"""
main_sell.py

Verkaufsskript für das Depot (Blatt 'portfolio', Spalte A).

Konzept (aus dem Konzeptgespräch):
  - Vier Bewertungskriterien, jedes ist gerissen (True), intakt (False) oder
    nicht auswertbar (None). Nicht auswertbare Kriterien zählen nicht.
      S1  KGV absolut > 25
      S2  KGV mehr als 20 % über dem eigenen 5-Jahres-Durchschnitt
          (Ø-KGV aus historischen Jahres-EPS, mindestens 3 Jahre)
      S3  Kurs mehr als doppelt so hoch wie der Graham-Preis
      S4  FCF-Rendite unter 3 % (bei Finanzwerten nicht auswertbar)
  - Verlust: Ist KGV oder Graham-Preis wegen negativem EPS/Buchwert nicht
    berechenbar, zählt das insgesamt als EIN gerissenes Kriterium
    (nicht je Bewertungskriterium).
  - Stufen: Warnung ab 2 gerissenen Kriterien, Verkauf ab 3. Die Schwellen
    bleiben absolut, auch wenn weniger als 4 Kriterien auswertbar sind.
  - Gekaufte Qualitätstitel können ein Verkaufssignal bekommen. Das ist
    gewollt, Konflikte prüft Pascal von Hand.
  - Ausgabe in ein neues Blatt 'Sell-signals' mit dem Layout des Kaufblatts.
    Das Depotblatt bleibt unverändert.

Fallback: Fehlt 'Diluted EPS' in der GuV, ist S2 nicht auswertbar und wird
markiert.
"""
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gspread
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
SCRIPT_VERSION = 'sell-v1-2026-10'

# --- SCHWELLENWERTE ---
S1_PE_MAX = 25.0
S2_REL_MAX = 0.20             # KGV mehr als 20 % über Ø
S2_MIN_JAHRE = 3
S3_GRAHAM_FACTOR = 2.0        # Kurs > 2x Graham-Preis
S4_FCF_YIELD_MIN = 3.0        # in Prozent; darunter gerissen
WARN_AB = 2
SELL_AB = 3

PAUSE_SECONDS = 1.5
RETRIES = 3
PE_HISTORY_YEARS = '5y'

PORTFOLIO_SHEET = 'portfolio'
OUTPUT_SHEET = 'Sell-signals'
FINANCIAL_SECTOR = 'Financial Services'

# Zeilennamen je Kennzahl (mehrere Kandidaten, siehe Label-Diagnose)
BS_LABELS: Dict[str, List[str]] = {
    'current_assets': ['Current Assets', 'Total Current Assets'],
    'current_liabilities': ['Current Liabilities', 'Total Current Liabilities'],
    'equity': ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'],
    'total_assets': ['Total Assets'],
}
FIN_LABELS: Dict[str, List[str]] = {
    'net_income': ['Net Income', 'Net Income Common Stockholders'],
    'diluted_eps': ['Diluted EPS', 'Basic EPS'],
}
CF_LABELS: Dict[str, List[str]] = {
    'free_cash_flow': ['Free Cash Flow'],
}

COLUMNS = [
    'Timestamp', 'Ticker', 'Name', 'Sektor', 'Finanzwert',
    'Kriterien_gerissen', 'Kriterien_bewertbar', 'Signal',
    'S1_KGV_Wert', 'S1_gerissen',
    'S2_Wert', 'S2_gerissen',
    'S3_Wert', 'S3_gerissen',
    'S4_Wert', 'S4_gerissen',
    'Verlust', 'Nicht_auswertbar',
    'Current_Price', 'PE_Ratio', 'PE_avg_5y', 'Graham_Price', 'FCF_Yield_%',
    'Market_Cap_Mrd', 'Datenfehler', 'Status', 'Version',
]


# --- HILFSFUNKTIONEN ---
def _num(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Spalten (Perioden) absteigend: iloc[0] = neueste Periode."""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        return df.sort_index(axis=1, ascending=False)
    except Exception:
        return df


def _find_row(df: pd.DataFrame, names: List[str]) -> Tuple[Optional[pd.Series], Optional[str]]:
    for name in names:
        if name in df.index:
            row = df.loc[name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row, name
    return None, None


def latest_value(df: pd.DataFrame, names: List[str]) -> Optional[float]:
    if df.empty:
        return None
    row, _ = _find_row(df, names)
    if row is None:
        return None
    return _num(row.iloc[0])


def series_values(df: pd.DataFrame, names: List[str]) -> pd.Series:
    """Alle verfügbaren Perioden, neueste zuerst."""
    if df.empty:
        return pd.Series(dtype=float)
    row, _ = _find_row(df, names)
    if row is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(row, errors='coerce').dropna()


class Kriterium:
    def __init__(self, gerissen: Optional[bool], wert: Optional[float] = None, grund: str = ''):
        self.gerissen = gerissen       # True = gerissen, False = intakt, None = nicht auswertbar
        self.wert = wert
        self.grund = grund

    @property
    def auswertbar(self) -> bool:
        return self.gerissen is not None


def fetch_ticker_data(ticker: str) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Holt info, GuV, Bilanz, Cashflow und Kursverlauf mit Wiederholungen."""
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRIES + 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            if not info:
                raise ValueError('info leer')
            hist = stock.history(period=PE_HISTORY_YEARS, interval='3mo')
            return info, stock.financials, stock.balance_sheet, stock.cashflow, hist
        except Exception as e:
            last_err = e
            wait = PAUSE_SECONDS * attempt * 2
            logger.warning(f"{ticker}: Versuch {attempt}/{RETRIES} fehlgeschlagen ({e}); warte {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"{ticker}: Abruf endgültig fehlgeschlagen ({last_err})")


def berechne_pe_avg_5y(hist: pd.DataFrame, eps_by_year: Dict[int, float]) -> Optional[float]:
    """
    5-Jahres-Durchschnitt des KGV: je Quartalskurs geteilt durch das EPS des
    Geschäftsjahres, das zum Kurszeitpunkt aktuell war. Jahre ohne positives
    EPS werden ausgelassen. Mindestens S2_MIN_JAHRE Jahre mit EPS nötig.
    """
    if hist is None or hist.empty or 'Close' not in hist or not eps_by_year:
        return None
    closes = hist['Close'].dropna()
    if closes.empty:
        return None
    pes = []
    for ts, close in closes.items():
        eps = eps_by_year.get(ts.year - 1) or eps_by_year.get(ts.year)
        if eps and eps > 0:
            pe = close / eps
            if 0 < pe < 500:
                pes.append(pe)
    if len(pes) < S2_MIN_JAHRE:
        return None
    return round(float(np.mean(pes)), 2)


def eps_je_jahr(financials: pd.DataFrame) -> Dict[int, float]:
    """Diluted EPS je Geschäftsjahr. Fehlt die Zeile, leeres Dict."""
    eps = series_values(financials, FIN_LABELS['diluted_eps'])
    out: Dict[int, float] = {}
    for ts, val in eps.items():
        jahr = pd.Timestamp(ts).year
        out[jahr] = float(val)
    return out


def get_sell_data(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        info, fin_raw, bs_raw, cf_raw, hist = fetch_ticker_data(ticker)
    except Exception as e:
        logger.error(f"{ticker}: {e}")
        return None

    name = info.get('longName', ticker)
    sector = info.get('sector')
    current_price = _num(info.get('currentPrice')) or _num(info.get('regularMarketPrice'))
    if not current_price or current_price <= 0:
        logger.warning(f"{ticker}: kein Kurs, übersprungen.")
        return None

    financials = _prepare(fin_raw)
    bs = _prepare(bs_raw)
    cf = _prepare(cf_raw)

    datenfehler: List[str] = []
    if financials.empty or bs.empty:
        datenfehler.append('Finanzdaten unvollständig')
    mcap = _num(info.get('marketCap'))
    if not mcap or mcap <= 0:
        datenfehler.append('Market Cap fehlt/0')

    # --- Finanzwert-Erkennung wie im Kaufskript ---
    ca = latest_value(bs, BS_LABELS['current_assets'])
    cl = latest_value(bs, BS_LABELS['current_liabilities'])
    is_financial = (ca is None or cl is None) and sector == FINANCIAL_SECTOR
    if (ca is None or cl is None) and sector != FINANCIAL_SECTOR and not bs.empty:
        datenfehler.append('Current Assets/Liabilities fehlen (kein Finanzsektor)')

    eps_ttm = _num(info.get('trailingEps'))
    book_value = _num(info.get('bookValue'))
    pe_now = _num(info.get('trailingPE'))
    pb = _num(info.get('priceToBook'))

    # --- Verlust-Erkennung: negatives EPS oder negativer Buchwert ---
    verlust = bool((eps_ttm is not None and eps_ttm <= 0) or (book_value is not None and book_value <= 0))

    kriterien: Dict[str, Kriterium] = {}

    # S1: KGV absolut
    if pe_now and pe_now > 0:
        kriterien['S1'] = Kriterium(pe_now > S1_PE_MAX, round(pe_now, 2))
    else:
        kriterien['S1'] = Kriterium(None, grund='KGV nicht berechenbar (Verlust)')

    # S2: KGV relativ zum eigenen 5-Jahres-Ø
    eps_hist = eps_je_jahr(financials)
    pe_avg = berechne_pe_avg_5y(hist, eps_hist)
    if pe_now and pe_now > 0 and pe_avg:
        rel = pe_now / pe_avg - 1
        kriterien['S2'] = Kriterium(rel > S2_REL_MAX, round(rel * 100, 1))
    else:
        grund = 'Historisches EPS fehlt oder zu wenige Jahre' if not pe_avg else 'KGV nicht berechenbar (Verlust)'
        kriterien['S2'] = Kriterium(None, grund=grund)

    # S3: Kurs vs. Graham-Preis
    graham_price = (22.5 * eps_ttm * book_value) ** 0.5 if (eps_ttm and eps_ttm > 0 and book_value and book_value > 0) else None
    if graham_price:
        kriterien['S3'] = Kriterium(current_price > S3_GRAHAM_FACTOR * graham_price, round(current_price / graham_price, 2))
    else:
        kriterien['S3'] = Kriterium(None, grund='Graham-Preis nicht berechenbar (Verlust)')

    # S4: FCF-Rendite
    fcf = latest_value(cf, CF_LABELS['free_cash_flow'])
    if is_financial:
        kriterien['S4'] = Kriterium(None, grund='Für Finanzwerte nicht anwendbar')
    elif fcf is not None and mcap and mcap > 0:
        fcf_yield = fcf / mcap * 100
        kriterien['S4'] = Kriterium(fcf_yield < S4_FCF_YIELD_MIN, round(fcf_yield, 2))
    else:
        kriterien['S4'] = Kriterium(None, grund='FCF nicht verfügbar')

    # --- Verlust zählt insgesamt als EIN Kriterium, nicht je Bewertungskriterium ---
    verlust_zaehlt = verlust
    auswertbare = [k for k in kriterien.values() if k.auswertbar]
    gerissen = sum(1 for k in auswertbare if k.gerissen)
    bewertbar = len(auswertbare)
    if verlust_zaehlt:
        gerissen += 1
        bewertbar += 1

    if gerissen >= SELL_AB:
        signal = 'VERKAUFEN'
    elif gerissen >= WARN_AB:
        signal = 'WARNUNG'
    else:
        signal = 'HALTEN'

    nicht_auswertbar = [f"{n}: {k.grund}" for n, k in kriterien.items() if not k.auswertbar and k.grund]

    return {
        'Timestamp': datetime.now().strftime('%d.%m.%Y %H:%M'),
        'Ticker': ticker,
        'Name': name,
        'Sektor': sector,
        'Finanzwert': is_financial,
        'Kriterien_gerissen': gerissen,
        'Kriterien_bewertbar': bewertbar,
        'Signal': signal,
        'S1_KGV_Wert': kriterien['S1'].wert,
        'S1_gerissen': kriterien['S1'].gerissen,
        'S2_Wert': kriterien['S2'].wert,
        'S2_gerissen': kriterien['S2'].gerissen,
        'S3_Wert': kriterien['S3'].wert,
        'S3_gerissen': kriterien['S3'].gerissen,
        'S4_Wert': kriterien['S4'].wert,
        'S4_gerissen': kriterien['S4'].gerissen,
        'Verlust': verlust,
        'Nicht_auswertbar': '; '.join(nicht_auswertbar) or None,
        'Current_Price': round(current_price, 2),
        'PE_Ratio': round(pe_now, 2) if pe_now else None,
        'PE_avg_5y': pe_avg,
        'Graham_Price': round(graham_price, 2) if graham_price else None,
        'FCF_Yield_%': kriterien['S4'].wert,
        'Market_Cap_Mrd': round(mcap / 1e9, 2) if mcap else None,
        'Datenfehler': '; '.join(datenfehler) or None,
        'Status': 'Datenfehler' if datenfehler else 'OK',
        'Version': SCRIPT_VERSION,
    }


def main() -> None:
    creds_json = os.getenv('GOOGLE_CREDS_JSON')
    if not creds_json:
        logger.critical("GOOGLE_CREDS_JSON nicht gefunden.")
        return
    if not SPREADSHEET_ID:
        logger.critical("SPREADSHEET_ID nicht gefunden.")
        return

    creds = Credentials.from_service_account_info(
        json.loads(creds_json), scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID)

    portfolio_ws = sheet.worksheet(PORTFOLIO_SHEET)
    tickers = list(dict.fromkeys(t.strip() for t in portfolio_ws.col_values(1)[1:] if t.strip()))

    rows: List[Dict[str, Any]] = []
    for t in tickers:
        logger.info(f"Prüfe Depot: {t}")
        data = get_sell_data(t)
        if data:
            rows.append(data)
        time.sleep(PAUSE_SECONDS)

    if not rows:
        logger.warning("Keine Daten berechnet, nichts geschrieben.")
        return

    def fmt(v: Any) -> Any:
        if v is None:
            return ''
        if isinstance(v, bool):
            return 'Ja' if v else 'Nein'
        return v

    output_rows = [[fmt(r.get(c)) for c in COLUMNS] for r in rows]

    # Ausgabeblatt anlegen, falls es noch nicht existiert
    try:
        output_ws = sheet.worksheet(OUTPUT_SHEET)
    except gspread.WorksheetNotFound:
        output_ws = sheet.add_worksheet(title=OUTPUT_SHEET, rows=1000, cols=len(COLUMNS))
        output_ws.append_row(COLUMNS, value_input_option='USER_ENTERED')

    # Batches statt Totalverlust bei einem Fehler
    geschrieben = 0
    for i in range(0, len(output_rows), 50):
        batch = output_rows[i:i + 50]
        try:
            output_ws.append_rows(batch, value_input_option='USER_ENTERED')
            geschrieben += len(batch)
        except Exception as e:
            logger.error(f"Batch {i}-{i + len(batch)} fehlgeschlagen ({e})")

    verkaufen = sum(1 for r in rows if r['Signal'] == 'VERKAUFEN')
    warnen = sum(1 for r in rows if r['Signal'] == 'WARNUNG')
    logger.info(f"Fertig: {geschrieben} von {len(output_rows)} Zeilen, {verkaufen} Verkauf, {warnen} Warnung.")


if __name__ == '__main__':
    main()
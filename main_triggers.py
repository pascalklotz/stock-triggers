"""
graham_score.py

Überarbeiteter Graham-Screener.

Konzept (siehe Konzeptgespräch):
  - Filter: mindestens 5 von maximal 6 auswertbaren Kriterien (K2-K7; K1 "Größe"
    wurde gestrichen). Nicht auswertbare Kriterien (fehlendes Datenlabel, zu
    wenig Jahre, o.ä.) fallen aus Zähler UND Nenner heraus und werden markiert,
    statt stillschweigend als "nicht erfüllt" zu zählen.
  - Finanzwerte (Banken, Versicherer, Vermögensverwalter) werden daran erkannt,
    dass Current Assets/Current Liabilities in der Bilanz fehlen UND der
    Yahoo-Sektor 'Financial Services' ist. Nur für K2/K3 gilt eine Ausnahme:
    K2 wird dort durch die Eigenkapitalquote ersetzt, K3 entfällt ganz.
  - Sortierung der Kandidaten (nur unter denen, die den Filter bestehen) über
    einen "Günstig-Rang": Perzentil-Mittelwert aus KGV, KBV, FCF-Rendite,
    Abstand zum eigenen 5-Jahres-Durchschnitts-KGV und Margin of Safety.
    Niedrigerer Rang = günstiger.
  - Bonus-Kriterien (Rückkäufe, FCF-Stabilität, ROIC) und die Margin of Safety
    sind reine Info-Spalten, kein Teil des Filters oder eines Scores mehr.

Kriterien im Detail:
  K2  Current Ratio >= 2                          (Nicht-Finanzwerte)
      Eigenkapitalquote >= 8%                     (Finanzwerte, Ersatz)
  K3  Nettoschulden / EBITDA <= 5                 (Nicht-Finanzwerte; für
                                                    Finanzwerte nicht anwendbar
                                                    und daher nicht auswertbar)
  K4  Net Income in allen verfügbaren Jahren > 0
  K5  Dividende (dividendYield > 0)
  K6  Lineare Regression der Jahresgewinne: Steigung / Durchschnittsgewinn
      >= 0.08. Nur auswertbar, wenn K4 erfüllt ist UND mindestens 3 Jahre
      vorliegen (sonst ist weder eine Gerade noch eine Aussage über
      "Stabilität" sinnvoll).
  K7  KGV <= 15 UND KBV <= 1.5 UND KGV*KBV <= 22.5

  - Netzwerkabruf mit Wiederholungen und Pausen zwischen Tickern (Rate-Limit-
    Schutz, siehe die Yahoo-Ausfälle in den Rohdaten aus dem Konzeptgespräch).
    Der 5-Jahres-Ø-KGV wird erst NACH dem Filter nur für die Kandidaten
    berechnet, die den Filter bestanden haben, um unnötige zusätzliche
    Netzwerkaufrufe zu vermeiden.
  B1  Sinkende Aktienzahl (Rückkäufe) - Label-Fix: 'Ordinary Shares Number'
      (der alte Code suchte fälschlich 'Ordinary Share Number', ohne 's')
  B2  Free Cash Flow in allen verfügbaren Jahren > 0
  B3  ROIC = Net Income / (Stockholders Equity + Total Debt) > 15%
      (unverändert, bekannte Näherung ohne NOPAT/Cash-Bereinigung)

Ausgabe: 22 Spalten (siehe COLUMNS), angehängt an das bestehende Blatt
'Buy-triggers'. Alte Zeilen (10 Spalten, Version leer) bleiben unangetastet;
die neuen Spalten sind bei ihnen einfach leer. Jede neue Zeile trägt in der
letzten Spalte die Skript-Version, damit alt und neu unterscheidbar bleiben.

Ausführen:
    pip install yfinance pandas gspread google-auth python-dotenv
    python graham_score.py
Voraussetzungen wie bisher: .env mit SPREADSHEET_ID, GOOGLE_CREDS_JSON als
Umgebungsvariable (GitHub Secret) mit den Service-Account-Zugangsdaten.
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

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
SCRIPT_VERSION = 'v2-2026-09'

# --- SCHWELLENWERTE (aus dem Konzeptgespräch, mit Daten aus der 40er-Stichprobe belegt) ---
CURRENT_RATIO_MIN = 2.0
NET_DEBT_EBITDA_MAX = 5.0
EQUITY_RATIO_MIN_FINANCIALS = 0.08          # Ersatzkriterium für K2 bei Finanzwerten
K6_GROWTH_MIN = 0.08                        # Steigung / Durchschnittsgewinn
K6_MIN_JAHRE = 3
K7_PE_MAX = 15.0
K7_PB_MAX = 1.5
K7_PE_TIMES_PB_MAX = 22.5
ROIC_MIN = 0.15
FILTER_MIN_ERFUELLT = 5                     # von max. 6 auswertbaren Kriterien
FINANCIAL_SECTOR = 'Financial Services'
PE_HISTORY_YEARS = '5y'
PAUSE_SECONDS = 1.5          # Pause zwischen Tickern (Rate-Limit-Schutz, wie im Verteilungsskript)
RETRIES = 3                  # Versuche pro Ticker bei Netzwerk-/Yahoo-Fehlern

# Mögliche Zeilennamen je Kennzahl (mehrere Kandidaten, da yfinance diese über
# Versionen hinweg geändert hat; siehe Label-Diagnose aus dem Konzeptgespräch).
BS_LABELS: Dict[str, List[str]] = {
    'current_assets': ['Current Assets', 'Total Current Assets'],
    'current_liabilities': ['Current Liabilities', 'Total Current Liabilities'],
    'total_debt': ['Total Debt'],
    'cash': ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents'],
    'equity': ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'],
    'total_assets': ['Total Assets'],
    'shares': ['Ordinary Shares Number', 'Share Issued'],   # Label-Fix ggü. Bestandscode
}
FIN_LABELS: Dict[str, List[str]] = {
    'ebitda': ['EBITDA', 'Normalized EBITDA'],
    'net_income': ['Net Income', 'Net Income Common Stockholders'],
}
CF_LABELS: Dict[str, List[str]] = {
    'free_cash_flow': ['Free Cash Flow'],
}

# Spaltenreihenfolge der Ausgabe (K = Spalte in Google Sheets, ab Spalte K neu)
COLUMNS = [
    'Timestamp', 'Ticker', 'Name', 'Sektor', 'Finanzwert',
    'Kriterien_erfuellt', 'Kriterien_bewertbar', 'Filter_bestanden',
    'K2_Wert', 'K2_erfuellt', 'K3_Wert', 'K3_erfuellt',
    'K4_erfuellt', 'K5_erfuellt', 'K6_Wert', 'K6_erfuellt',
    'K7_erfuellt', 'Nicht_auswertbar',
    'Current_Price', 'PE_Ratio', 'PB_Ratio', 'PE_avg_5y', 'FCF_Yield_%',
    'Margin_of_Safety_%', 'Guenstig_Rang',
    'Bonus_Rueckkaeufe', 'Bonus_FCF_stabil', 'Bonus_ROIC_ueber_15%',
    'Market_Cap_Mrd', 'Datenfehler', 'Status', 'Version',
]


# --- HILFSFUNKTIONEN ---
def _num(value: Any) -> Optional[float]:
    """Wandelt None/NaN/Nicht-Zahlen sicher in float oder None um."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Sortiert die Spalten (Perioden) absteigend: iloc[0] = neueste Periode."""
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
    """Alle verfügbaren (nicht-leeren) Perioden, neueste zuerst."""
    if df.empty:
        return pd.Series(dtype=float)
    row, _ = _find_row(df, names)
    if row is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(row, errors='coerce').dropna()


class Kriterium:
    """Ergebnis eines einzelnen Kriteriums: erfüllt / nicht erfüllt / nicht auswertbar."""

    def __init__(self, erfuellt: Optional[bool], wert: Optional[float] = None, grund: str = ''):
        self.erfuellt = erfuellt          # None = nicht auswertbar
        self.wert = wert
        self.grund = grund

    @property
    def auswertbar(self) -> bool:
        return self.erfuellt is not None


def berechne_pe_avg_5y(ticker_symbol: str, trailing_eps: Optional[float]) -> Optional[float]:
    """
    Näherungsweiser 5-Jahres-Durchschnitt des KGV: historische Schlusskurse
    in Quartalsabständen (ca. 20 Datenpunkte über 5 Jahre) geteilt durch das
    AKTUELLE trailing EPS. Das ist eine bewusste Vereinfachung (siehe
    Konzeptgespräch): yfinance liefert keine historischen
    EPS-Reihen im Standardzugriff, daher wird das heutige EPS als Näherung für
    die Vergangenheit verwendet. Bei stark schwankenden Gewinnen ist dieser
    Wert entsprechend ungenau - das gehört als Hinweis in die Doku, nicht ins
    Kriterium selbst (Ø-KGV ist nur eine Sortierhilfe, kein Filterkriterium).
    """
    if not trailing_eps or trailing_eps <= 0:
        return None
    try:
        hist = yf.Ticker(ticker_symbol).history(period=PE_HISTORY_YEARS, interval='3mo')
        if hist is None or hist.empty or 'Close' not in hist:
            return None
        closes = hist['Close'].dropna()
        if closes.empty:
            return None
        pe_series = closes / trailing_eps
        pe_series = pe_series[(pe_series > 0) & (pe_series < 500)]  # Ausreißer/Fehlwerte kappen
        if pe_series.empty:
            return None
        return round(float(pe_series.mean()), 2)
    except Exception as e:
        logger.warning(f"{ticker_symbol}: Ø-KGV nicht berechenbar ({e})")
        return None


def fetch_ticker_data(ticker_symbol: str) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Holt info/financials/balance_sheet/cashflow mit Wiederholungen bei Fehlern."""
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRIES + 1):
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info or {}
            if not info:
                raise ValueError('info leer')
            return info, stock.financials, stock.balance_sheet, stock.cashflow
        except Exception as e:
            last_err = e
            wait = PAUSE_SECONDS * attempt * 2
            logger.warning(f"{ticker_symbol}: Versuch {attempt}/{RETRIES} fehlgeschlagen ({e}); warte {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"{ticker_symbol}: Abruf endgültig fehlgeschlagen ({last_err})")


def get_graham_data(ticker_symbol: str) -> Optional[Dict[str, Any]]:
    """Holt alle Rohdaten und berechnet Kriterien, Kennzahlen und Bonus-Infos."""
    try:
        info, financials_raw, bs_raw, cf_raw = fetch_ticker_data(ticker_symbol)

        name = info.get('longName', ticker_symbol)
        sector = info.get('sector')

        current_price = _num(info.get('currentPrice')) or _num(info.get('regularMarketPrice'))
        if not current_price or current_price <= 0:
            logger.warning(f"{ticker_symbol}: kein Kurs verfügbar, überspringe.")
            return None

        financials = _prepare(financials_raw)
        bs = _prepare(bs_raw)
        cf = _prepare(cf_raw)

        row: Dict[str, Any] = {
            'Timestamp': datetime.now().strftime('%d.%m.%Y %H:%M'),
            'Ticker': ticker_symbol,
            'Name': name,
            'Sektor': sector,
            'Current_Price': round(current_price, 2),
            'Version': SCRIPT_VERSION,
        }

        datenfehler: List[str] = []
        if financials.empty or bs.empty:
            datenfehler.append('Finanzdaten unvollständig (GuV/Bilanz leer)')

        mcap = _num(info.get('marketCap'))
        if not mcap or mcap <= 0:
            datenfehler.append('Market Cap fehlt/0')
        row['Market_Cap_Mrd'] = round(mcap / 1e9, 2) if mcap else None

        # --- Bilanzposten ---
        ca = latest_value(bs, BS_LABELS['current_assets'])
        cl = latest_value(bs, BS_LABELS['current_liabilities'])
        debt = latest_value(bs, BS_LABELS['total_debt'])
        cash = latest_value(bs, BS_LABELS['cash'])
        equity = latest_value(bs, BS_LABELS['equity'])
        total_assets = latest_value(bs, BS_LABELS['total_assets'])

        # --- Finanzwert-Erkennung: Current Assets/Liabilities fehlen UND Sektor Financial Services ---
        missing_ca_cl = ca is None or cl is None
        is_financial = missing_ca_cl and sector == FINANCIAL_SECTOR
        row['Finanzwert'] = is_financial
        if missing_ca_cl and sector != FINANCIAL_SECTOR and not bs.empty:
            datenfehler.append('Current Assets/Liabilities fehlen (kein Finanzsektor)')

        kriterien: Dict[str, Kriterium] = {}

        # --- K2: Current Ratio (Nicht-Finanzwerte) / Eigenkapitalquote (Finanzwerte) ---
        if is_financial:
            if equity is not None and total_assets and total_assets > 0:
                eq_ratio = equity / total_assets
                kriterien['K2'] = Kriterium(eq_ratio >= EQUITY_RATIO_MIN_FINANCIALS, round(eq_ratio * 100, 1))
            else:
                kriterien['K2'] = Kriterium(None, grund='Eigenkapitalquote nicht berechenbar')
        else:
            if ca is not None and cl and cl > 0:
                cr = ca / cl
                kriterien['K2'] = Kriterium(cr >= CURRENT_RATIO_MIN, round(cr, 2))
            else:
                kriterien['K2'] = Kriterium(None, grund='Current Ratio nicht berechenbar')

        # --- K3: Nettoschulden / EBITDA (nur Nicht-Finanzwerte; für Finanzwerte nicht sinnvoll) ---
        if is_financial:
            kriterien['K3'] = Kriterium(None, grund='Für Finanzwerte nicht anwendbar')
        else:
            ebitda = latest_value(financials, FIN_LABELS['ebitda'])
            if debt is not None and cash is not None and ebitda is not None:
                net_debt = debt - cash
                if net_debt <= 0:
                    kriterien['K3'] = Kriterium(True, round(net_debt / 1e9, 2))  # nettoschuldenfrei
                elif ebitda > 0:
                    ratio = net_debt / ebitda
                    kriterien['K3'] = Kriterium(ratio <= NET_DEBT_EBITDA_MAX, round(ratio, 2))
                else:
                    kriterien['K3'] = Kriterium(False, grund='EBITDA <= 0 bei bestehenden Nettoschulden')
            else:
                kriterien['K3'] = Kriterium(None, grund='Nettoschulden/EBITDA nicht berechenbar')

        # --- K4/K6: Gewinne ---
        net_income = series_values(financials, FIN_LABELS['net_income'])
        alle_positiv = bool((net_income > 0).all()) if len(net_income) > 0 else None
        if len(net_income) == 0:
            kriterien['K4'] = Kriterium(None, grund='Net Income nicht verfügbar')
        else:
            kriterien['K4'] = Kriterium(alle_positiv)

        if len(net_income) >= K6_MIN_JAHRE and alle_positiv:
            chron = net_income.iloc[::-1]  # chronologisch aufsteigend für die Regression
            x = np.arange(len(chron))
            slope, _ = np.polyfit(x, chron.values, 1)
            mean_ni = float(chron.mean())
            if mean_ni != 0:
                kennzahl = float(slope) / mean_ni
                kriterien['K6'] = Kriterium(kennzahl >= K6_GROWTH_MIN, round(kennzahl, 4))
            else:
                kriterien['K6'] = Kriterium(None, grund='Durchschnittsgewinn = 0')
        else:
            grund = 'weniger als 3 Jahre' if len(net_income) < K6_MIN_JAHRE else 'K4 nicht erfüllt'
            kriterien['K6'] = Kriterium(None, grund=f'Wachstum nicht bewertbar ({grund})')

        # --- K5: Dividende ---
        dividend_yield = _num(info.get('dividendYield'))
        kriterien['K5'] = Kriterium((dividend_yield or 0) > 0)

        # --- K7: Bewertung (KGV, KBV) ---
        pe = _num(info.get('trailingPE'))
        pb = _num(info.get('priceToBook'))
        if pe and pe > 0 and pb and pb > 0:
            k7_ok = pe <= K7_PE_MAX and pb <= K7_PB_MAX and (pe * pb) <= K7_PE_TIMES_PB_MAX
            kriterien['K7'] = Kriterium(k7_ok)
        else:
            kriterien['K7'] = Kriterium(None, grund='KGV/KBV nicht verfügbar')

        # --- Filter auswerten: mind. 5 von max. 6 AUSWERTBAREN Kriterien ---
        auswertbare = [k for k in kriterien.values() if k.auswertbar]
        erfuellte = [k for k in auswertbare if k.erfuellt]
        row['Kriterien_erfuellt'] = len(erfuellte)
        row['Kriterien_bewertbar'] = len(auswertbare)
        row['Filter_bestanden'] = len(erfuellte) >= FILTER_MIN_ERFUELLT
        row['Nicht_auswertbar'] = ', '.join(
            f"{name}: {k.grund}" for name, k in kriterien.items() if not k.auswertbar and k.grund
        ) or None

        for kname in ['K2', 'K3', 'K4', 'K5', 'K6', 'K7']:
            k = kriterien[kname]
            if kname in ('K2', 'K3', 'K6'):
                row[f'{kname}_Wert'] = k.wert
            row[f'{kname}_erfuellt'] = k.erfuellt  # None bleibt None (nicht auswertbar)

        # --- Graham Price / Margin of Safety (Info, kein Filterkriterium) ---
        eps = _num(info.get('trailingEps'))
        book_value = _num(info.get('bookValue'))
        graham_price = (22.5 * eps * book_value) ** 0.5 if (eps and eps > 0 and book_value and book_value > 0) else None
        margin_of_safety = (1 - current_price / graham_price) * 100 if graham_price else None
        row['Margin_of_Safety_%'] = round(margin_of_safety, 1) if margin_of_safety is not None else None
        row['PE_Ratio'] = round(pe, 2) if pe else None
        row['PB_Ratio'] = round(pb, 2) if pb else None

        # --- FCF-Rendite ---
        fcf = latest_value(cf, CF_LABELS['free_cash_flow'])
        fcf_yield = (fcf / mcap * 100) if (fcf is not None and mcap and mcap > 0) else None
        row['FCF_Yield_%'] = round(fcf_yield, 2) if fcf_yield is not None else None

        # --- 5-Jahres-Ø-KGV: wird erst NACH dem Filter für die Besteher berechnet
        # (siehe ergaenze_pe_avg_5y), um nicht für jeden Ticker unnötig einen
        # zusätzlichen .history()-Aufruf zu machen ---
        # --- 5-Jahres-Ø-KGV: wird erst NACH dem Filter für die Besteher berechnet
        # (siehe ergaenze_pe_avg_5y), um nicht für jeden Ticker unnötig einen
        # zusätzlichen .history()-Aufruf zu machen ---
        row['PE_avg_5y'] = None
        row['_trailing_eps'] = eps  # intern für ergaenze_pe_avg_5y, wird vor der Ausgabe entfernt

        # --- Bonus 1: Rückkäufe (Label-Fix) ---
        shares = series_values(bs, BS_LABELS['shares'])
        if len(shares) > 1:
            row['Bonus_Rueckkaeufe'] = bool(shares.iloc[0] < shares.iloc[1])
        else:
            row['Bonus_Rueckkaeufe'] = None

        # --- Bonus 2: FCF in allen verfügbaren Jahren positiv ---
        fcf_series = series_values(cf, CF_LABELS['free_cash_flow'])
        row['Bonus_FCF_stabil'] = bool((fcf_series > 0).all()) if len(fcf_series) > 0 else None

        # --- Bonus 3: ROIC (unverändert, bekannte Näherung) ---
        if equity is not None and debt is not None and len(net_income) > 0:
            invested_capital = equity + debt
            if invested_capital > 0:
                roic = net_income.iloc[0] / invested_capital
                row['Bonus_ROIC_ueber_15%'] = bool(roic > ROIC_MIN)
            else:
                row['Bonus_ROIC_ueber_15%'] = None
        else:
            row['Bonus_ROIC_ueber_15%'] = None

        row['Datenfehler'] = '; '.join(datenfehler) if datenfehler else None
        row['Status'] = 'Datenfehler' if datenfehler else 'OK'
        row['Guenstig_Rang'] = None  # wird nach dem Sammeln aller Zeilen über die Kandidaten berechnet

        return row

    except Exception as e:
        logger.error(f"Fehler bei {ticker_symbol}: {e}", exc_info=True)
        return None


def ergaenze_pe_avg_5y(rows: List[Dict[str, Any]]) -> None:
    """
    Berechnet PE_avg_5y nur für Ticker, die den Filter bestanden haben (spart
    einen zusätzlichen .history()-Aufruf pro Ticker für alle, die ohnehin
    nicht in die Sortierung einfließen). Ändert 'rows' in place und entfernt
    danach das interne '_trailing_eps'-Feld aus allen Zeilen.
    """
    for r in rows:
        eps = r.pop('_trailing_eps', None)
        if r.get('Filter_bestanden'):
            r['PE_avg_5y'] = berechne_pe_avg_5y(r['Ticker'], eps)
            time.sleep(PAUSE_SECONDS)  # zusätzlicher Netzwerkaufruf, gleicher Rate-Limit-Schutz


def berechne_guenstig_rang(rows: List[Dict[str, Any]]) -> None:
    """
    Berechnet für alle Zeilen, die den Filter bestanden haben, einen
    Perzentil-Rang aus KGV, KBV, FCF-Rendite (invertiert, da höher besser),
    Abstand zum eigenen Ø-KGV und Margin of Safety (invertiert, da höher
    besser). Niedrigerer Rang = günstiger. Ändert 'rows' in place.
    """
    kandidaten = [r for r in rows if r.get('Filter_bestanden')]
    if not kandidaten:
        return

    df = pd.DataFrame(kandidaten)
    df['Abstand_Ø_KGV'] = df['PE_Ratio'] - df['PE_avg_5y']

    # Perzentilrang je Kennzahl (0=günstigster, 1=teuerster). FCF-Rendite und
    # Margin of Safety sind "höher = besser", daher hier invertiert (1 - rank),
    # damit in allen fünf Spalten "niedriger = günstiger" gilt.
    def rank01(series: pd.Series, invert: bool) -> pd.Series:
        r = series.rank(pct=True, na_option='keep')
        return (1 - r) if invert else r

    komponenten = pd.DataFrame({
        'r_pe': rank01(df['PE_Ratio'], invert=False),
        'r_pb': rank01(df['PB_Ratio'], invert=False),
        'r_fcf': rank01(df['FCF_Yield_%'], invert=True),
        'r_abstand': rank01(df['Abstand_Ø_KGV'], invert=False),
        'r_mos': rank01(df['Margin_of_Safety_%'], invert=True),
    })
    df['Guenstig_Rang'] = komponenten.mean(axis=1, skipna=True).round(4)

    rang_by_ticker = dict(zip(df['Ticker'], df['Guenstig_Rang']))
    for r in rows:
        if r['Ticker'] in rang_by_ticker:
            r['Guenstig_Rang'] = rang_by_ticker[r['Ticker']]


def main() -> None:
    creds_json = os.getenv('GOOGLE_CREDS_JSON')
    if not creds_json:
        logger.critical("Fehler: GOOGLE_CREDS_JSON nicht gefunden.")
        return
    if not SPREADSHEET_ID:
        logger.critical("Fehler: SPREADSHEET_ID nicht gefunden.")
        return

    creds_dict = json.loads(creds_json)
    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    sheet = gc.open_by_key(SPREADSHEET_ID)
    input_ws = sheet.worksheet('stocklist')
    output_ws = sheet.worksheet('advanced buy triggers')

    tickers = list(dict.fromkeys(t.strip() for t in input_ws.col_values(1)[1:] if t.strip()))

    rows: List[Dict[str, Any]] = []
    for symbol in tickers:
        logger.info(f"Verarbeite: {symbol}")
        try:
            data = get_graham_data(symbol)
        except Exception as e:
            logger.error(f"{symbol}: endgültig fehlgeschlagen, überspringe ({e})")
            data = None
        if data:
            rows.append(data)
        time.sleep(PAUSE_SECONDS)

    if not rows:
        logger.warning("Keine Daten berechnet, nichts geschrieben.")
        return

    ergaenze_pe_avg_5y(rows)
    berechne_guenstig_rang(rows)

    # Auf feste Spaltenreihenfolge bringen; Booleans als deutsche Texte, damit
    # sie im Sheet lesbar sind (statt TRUE/FALSE bzw. Python-Repr).
    def fmt(value: Any) -> Any:
        if value is None:
            return ''
        if isinstance(value, bool):
            return 'Ja' if value else 'Nein'
        return value

    output_rows = [[fmt(r.get(col)) for col in COLUMNS] for r in rows]

    try:
        output_ws.append_rows(output_rows, value_input_option='USER_ENTERED')
    except Exception as e:
        # Nicht alles verlieren: in Batches nachschreiben, damit ein einzelnes
        # Limit/Fehler nicht die gesamte Ausbeute des Laufs vernichtet.
        logger.error(f"append_rows für alle {len(output_rows)} Zeilen fehlgeschlagen ({e}); versuche Batches.")
        geschrieben = 0
        batch_size = 50
        for i in range(0, len(output_rows), batch_size):
            batch = output_rows[i:i + batch_size]
            try:
                output_ws.append_rows(batch, value_input_option='USER_ENTERED')
                geschrieben += len(batch)
            except Exception as e2:
                logger.error(f"Batch {i}-{i+len(batch)} fehlgeschlagen, übersprungen ({e2})")
        logger.warning(f"Nach Batch-Wiederholung: {geschrieben} von {len(output_rows)} Zeilen geschrieben.")

    bestanden = sum(1 for r in rows if r.get('Filter_bestanden'))
    logger.info(f"Fertig: {len(rows)} Ticker verarbeitet, {bestanden} davon haben den Filter bestanden.")


if __name__ == '__main__':
    main()
"""
kriterien_verteilung.py

Zweck: Verteilung der geplanten Graham-Kriterien über eine Stichprobe von
40 Tickern ermitteln, damit Schwellenwerte aus Daten statt aus dem Bauch
festgelegt werden können.

Ermittelt je Ticker:
  - Sektor / Branche (Yahoo)
  - Finanzwert-Erkennung: Current Assets/Liabilities fehlen UND Sektor = 'Financial Services'
  - Current Ratio                      (Kriterium 2, nur Nicht-Finanzwerte)
  - Nettoschulden / EBITDA             (neues Kriterium 3, nur Nicht-Finanzwerte)
  - Eigenkapitalquote                  (Ersatzkriterium für Finanzwerte)
  - Gewinnjahre, Gewinnstabilität                     (Kriterium 4)
  - Gewinnwachstum als Regressionssteigung / Durchschnittsgewinn (Kriterium 6, NEU)
  - Dividende ja/nein                  (Kriterium 5)
  - Datenqualität: fehlende Posten, Market Cap = 0, leere Abfragen

Kriterium 6 (NEU, ersetzt den alten Randjahr-Vergleich):
  Lineare Regression der Jahresgewinne (x = Jahresindex, y = Net Income).
  Kennzahl = Steigung / Durchschnittsgewinn der Periode (Wachstum pro Jahr,
  relativ zum Gewinnniveau). Nur bewertet, wenn zusätzlich ALLE verfügbaren
  Jahre positiv sind (wie Kriterium 4) und mindestens 3 Jahre vorliegen
  (eine Gerade durch 2 Punkte ist kein Trend). Der Schwellenwert für die
  Kennzahl ist noch offen; dieses Skript gibt seine Verteilung aus, damit
  er aus echten Daten statt aus dem Bauch festgelegt werden kann.

Zusätzlich eine Label-Diagnose: welche Bilanz-/GuV-Zeilennamen Yahoo bei dir
tatsächlich liefert. Damit lässt sich prüfen, ob der Bestandscode seine Posten
(z. B. 'Total Current Assets') überhaupt findet.

Ausführen (lokal, Internetzugang zu Yahoo nötig):
    pip install yfinance pandas
    python kriterien_verteilung.py
Ergebnis: Auswertung in der Konsole und die Datei kriterien_verteilung.csv
(Trennzeichen ';', Dezimalkomma, für deutsches Excel/Sheets).
"""
import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- KONFIGURATION ---
# Stichprobe: 40 Ticker, gezogen aus den 297 Tickern der Rohdaten (random.Random(42).sample)
SAMPLE: List[str] = [
    'ABT', 'ALB', 'ALGN', 'ALLE', 'AMCR', 'ARE', 'BSX', 'CARR', 'CBRE', 'CDNS',
    'CEG', 'CI', 'CMS', 'CRI', 'CVX', 'D', 'EQIX', 'FE', 'FFIN', 'FHN',
    'FITB', 'FTNT', 'GNTX', 'HP', 'ICE', 'ILMN', 'MCD', 'MCK', 'MET', 'MOS',
    'NOV', 'NTRS', 'PH', 'PLD', 'RJF', 'RSG', 'TDY', 'UPS', 'VMC', 'WMB',
]

PAUSE_SECONDS = 1.5          # Pause zwischen Tickern (Rate-Limit-Schutz)
RETRIES = 3                  # Versuche pro Ticker
OUTPUT_CSV = 'kriterien_verteilung.csv'
K6_MIN_JAHRE = 3              # Kriterium 6 (NEU): mindestens 3 Jahre für eine sinnvolle Regressionsgerade
THRESHOLDS_ND_EBITDA = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
FINANCIAL_SECTOR = 'Financial Services'

# Mögliche Zeilennamen je Kennzahl (Reihenfolge = Priorität). Yahoo hat die Namen
# über yfinance-Versionen hinweg geändert, deshalb mehrere Kandidaten.
BS_LABELS: Dict[str, List[str]] = {
    'current_assets': ['Current Assets', 'Total Current Assets'],
    'current_liabilities': ['Current Liabilities', 'Total Current Liabilities'],
    'total_debt': ['Total Debt'],
    'long_term_debt': ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'],
    'cash': ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents'],
    'net_debt': ['Net Debt'],
    'equity': ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest'],
    'total_assets': ['Total Assets'],
}
FIN_LABELS: Dict[str, List[str]] = {
    'ebitda': ['EBITDA', 'Normalized EBITDA'],
    'net_income': ['Net Income', 'Net Income Common Stockholders'],
}


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
            if isinstance(row, pd.DataFrame):   # doppelte Zeilennamen absichern
                row = row.iloc[0]
            return row, name
    return None, None


def latest_value(df: pd.DataFrame, names: List[str]) -> Tuple[Optional[float], Optional[str]]:
    """Wert der neuesten Periode und der Zeilenname, der getroffen hat (None, None wenn fehlt)."""
    if df.empty:
        return None, None
    row, label = _find_row(df, names)
    if row is None:
        return None, None
    value = _num(row.iloc[0])
    return (value, label) if value is not None else (None, None)


def series_values(df: pd.DataFrame, names: List[str]) -> pd.Series:
    """Alle verfügbaren (nicht-leeren) Perioden, neueste zuerst."""
    if df.empty:
        return pd.Series(dtype=float)
    row, _ = _find_row(df, names)
    if row is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(row, errors='coerce').dropna()


# --- KENNZAHLEN JE TICKER ---
def compute_metrics(ticker: str, info: Dict[str, Any], financials: Optional[pd.DataFrame],
                    balance_sheet: Optional[pd.DataFrame]) -> Tuple[Dict[str, Any], Dict[str, Optional[str]]]:
    """Berechnet alle Kennzahlen eines Tickers. Gibt (Zeile, verwendete Zeilennamen) zurück."""
    fin = _prepare(financials)
    bal = _prepare(balance_sheet)
    flags: List[str] = []
    labels_used: Dict[str, Optional[str]] = {}

    sector = info.get('sector')
    mcap = _num(info.get('marketCap'))
    row: Dict[str, Any] = {
        'Ticker': ticker,
        'Name': info.get('longName') or ticker,
        'Sektor': sector,
        'Branche': info.get('industry'),
        'MarketCap_Mrd': round(mcap / 1e9, 2) if mcap else None,
    }
    if not mcap or mcap <= 0:
        flags.append('Market Cap fehlt/0')

    daten_ok = not fin.empty and not bal.empty
    row['Daten_ok'] = daten_ok
    if not daten_ok:
        flags.append('Finanzdaten fehlen')

    # --- Bilanzposten ---
    ca, labels_used['current_assets'] = latest_value(bal, BS_LABELS['current_assets'])
    cl, labels_used['current_liabilities'] = latest_value(bal, BS_LABELS['current_liabilities'])
    debt, labels_used['total_debt'] = latest_value(bal, BS_LABELS['total_debt'])
    _, labels_used['long_term_debt'] = latest_value(bal, BS_LABELS['long_term_debt'])
    cash, labels_used['cash'] = latest_value(bal, BS_LABELS['cash'])
    net_debt_row, labels_used['net_debt'] = latest_value(bal, BS_LABELS['net_debt'])
    equity, labels_used['equity'] = latest_value(bal, BS_LABELS['equity'])
    total_assets, labels_used['total_assets'] = latest_value(bal, BS_LABELS['total_assets'])

    # --- Finanzwert-Erkennung: Posten fehlen UND Sektor = Financial Services ---
    missing_ca_cl = ca is None or cl is None
    sector_financial = sector == FINANCIAL_SECTOR
    is_financial = daten_ok and missing_ca_cl and sector_financial
    row['Sektor_Finanz'] = sector_financial
    row['Finanzwert'] = is_financial
    if daten_ok and missing_ca_cl and not sector_financial:
        flags.append('Datenfehler: Current Assets/Liabilities fehlen (kein Finanzsektor)')

    # --- Kriterium 2: Current Ratio ---
    row['Current_Ratio'] = round(ca / cl, 2) if (ca is not None and cl and cl > 0) else None

    # --- Neues Kriterium 3: Nettoschulden / EBITDA ---
    if debt is not None and cash is not None:
        net_debt: Optional[float] = debt - cash
    else:
        net_debt = net_debt_row
    ebitda, labels_used['ebitda'] = latest_value(fin, FIN_LABELS['ebitda'])
    row['Nettoschulden_Mrd'] = round(net_debt / 1e9, 2) if net_debt is not None else None
    row['EBITDA_Mrd'] = round(ebitda / 1e9, 2) if ebitda is not None else None
    row['Nettoschuldenfrei'] = (net_debt <= 0) if net_debt is not None else None
    row['EBITDA_nicht_positiv'] = (ebitda <= 0) if ebitda is not None else None
    if net_debt is not None and ebitda is not None and ebitda > 0:
        row['NettoSchulden_EBITDA'] = round(net_debt / ebitda, 2)
    else:
        row['NettoSchulden_EBITDA'] = None

    # --- Ersatzkriterium Finanzwerte: Eigenkapitalquote ---
    if equity is not None and total_assets and total_assets > 0:
        row['Eigenkapitalquote_%'] = round(equity / total_assets * 100, 1)
    else:
        row['Eigenkapitalquote_%'] = None

    # --- Kriterium 4: Gewinnstabilität ---
    net_income = series_values(fin, FIN_LABELS['net_income'])
    _, labels_used['net_income'] = latest_value(fin, FIN_LABELS['net_income'])
    row['Gewinnjahre'] = int(len(net_income))
    alle_positiv = bool((net_income > 0).all()) if len(net_income) > 0 else None
    row['Gewinne_alle_positiv'] = alle_positiv

    # --- Kriterium 6 (NEU): Regressionssteigung / Durchschnittsgewinn ---
    # net_income ist neueste zuerst (series_values sortiert wie die Bilanz/GuV-Spalten
    # absteigend); für die Regression drehen wir auf chronologisch aufsteigend.
    if len(net_income) >= K6_MIN_JAHRE:
        chron = net_income.iloc[::-1]
        x = np.arange(len(chron))
        slope, _intercept = np.polyfit(x, chron.values, 1)
        mean_ni = float(chron.mean())
        if mean_ni != 0:
            row['K6_Steigung_pro_Jahr'] = round(float(slope), 2)
            row['K6_Durchschnittsgewinn'] = round(mean_ni, 2)
            row['K6_Kennzahl'] = round(float(slope) / mean_ni, 4)
        else:
            row['K6_Steigung_pro_Jahr'] = None
            row['K6_Durchschnittsgewinn'] = None
            row['K6_Kennzahl'] = None
    else:
        row['K6_Steigung_pro_Jahr'] = None
        row['K6_Durchschnittsgewinn'] = None
        row['K6_Kennzahl'] = None
    # K6 nur bewertbar, wenn genug Jahre da sind UND alle positiv sind (wie K4)
    row['K6_bewertbar'] = bool(len(net_income) >= K6_MIN_JAHRE and alle_positiv is True)

    # --- Kriterium 5: Dividende ---
    row['Dividende'] = (_num(info.get('dividendYield')) or 0) > 0

    row['Status'] = 'OK' if not flags else '; '.join(flags)
    return row, labels_used


# --- DATENABRUF ---
def fetch(ticker: str) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Holt Info, GuV und Bilanz mit Wiederholungen bei Fehlern."""
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRIES + 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            if not info:
                raise ValueError('info leer')
            return info, stock.financials, stock.balance_sheet
        except Exception as e:
            last_err = e
            wait = PAUSE_SECONDS * attempt * 2
            logger.warning(f"{ticker}: Versuch {attempt}/{RETRIES} fehlgeschlagen ({e}); warte {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"{ticker}: Abruf endgültig fehlgeschlagen ({last_err})")


# --- AUSWERTUNG ---
def _percentiles(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return 'keine Werte'
    qs = {f'P{int(q * 100)}': round(float(s.quantile(q)), 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return ', '.join(f'{k}={v}' for k, v in qs.items()) + f' (n={len(s)})'


def print_summary(df: pd.DataFrame, label_counter: Dict[str, Counter], n_total: int, failed: List[str]) -> None:
    pd.set_option('display.width', 220, 'display.max_columns', 30, 'display.max_rows', 100)
    line = '=' * 78

    print(f'\n{line}\n1) DATENABDECKUNG\n{line}')
    print(f'Ticker in Stichprobe: {n_total} | erfolgreich abgerufen: {len(df)} | fehlgeschlagen: {len(failed)}')
    if failed:
        print('Fehlgeschlagen:', ', '.join(failed))
    print(f'Mit vollständiger GuV und Bilanz: {int(df["Daten_ok"].sum())} von {len(df)}')
    bad = df[df['Status'] != 'OK']
    if not bad.empty:
        print('\nTicker mit Auffälligkeiten:')
        print(bad[['Ticker', 'Sektor', 'Status']].to_string(index=False))

    print(f'\n{line}\n2) LABEL-DIAGNOSE (welche Zeilennamen liefert Yahoo tatsächlich?)\n{line}')
    ok_n = int(df['Daten_ok'].sum())
    for key, counter in label_counter.items():
        found = sum(counter.values())
        detail = ', '.join(f"'{name}': {n}" for name, n in counter.most_common()) or 'nie gefunden'
        print(f'  {key:<22} gefunden bei {found:>2} von {ok_n} | {detail}')
    tca = label_counter['current_assets'].get('Total Current Assets', 0)
    ca_new = label_counter['current_assets'].get('Current Assets', 0)
    if tca == 0 and ca_new > 0:
        print("\n  HINWEIS: 'Total Current Assets' wird nie getroffen, 'Current Assets' schon.")
        print("  Der Bestandscode sucht nur 'Total Current Assets'. Wenn das stimmt, waren die")
        print("  Kriterien 2 und 3 dort bisher praktisch nie erfüllbar (Werte 0 -> Bedingungen falsch).")

    print(f'\n{line}\n3) SEKTOREN UND FINANZWERT-ERKENNUNG\n{line}')
    print(df['Sektor'].value_counts(dropna=False).to_string())
    print(f'\nAls Finanzwert erkannt (Posten fehlen UND Financial Services): {int(df["Finanzwert"].sum())}')
    print(f'Sektor Financial Services insgesamt: {int(df["Sektor_Finanz"].sum())}')
    fin_sector_but_normal = df[df['Sektor_Finanz'] & ~df['Finanzwert']]
    if not fin_sector_but_normal.empty:
        print('Financial Services, aber Current Assets/Liabilities vorhanden (werden wie Nicht-Finanzwerte behandelt):')
        print('  ' + ', '.join(fin_sector_but_normal['Ticker']))

    base = df[df['Daten_ok'] & ~df['Finanzwert']]

    print(f'\n{line}\n4) NETTOSCHULDEN / EBITDA (Nicht-Finanzwerte, n={len(base)})\n{line}')
    ratio = base['NettoSchulden_EBITDA']
    print('Verteilung (nur EBITDA > 0; negative Werte = Nettobarmittel):', _percentiles(ratio))
    nettofrei = int((base['Nettoschuldenfrei'] == True).sum())          # noqa: E712
    ebitda_neg = int(((base['EBITDA_nicht_positiv'] == True) & (base['Nettoschuldenfrei'] != True)).sum())  # noqa: E712
    nicht_messbar = int((base['NettoSchulden_EBITDA'].isna() & base['Nettoschuldenfrei'].isna()).sum())
    print(f'Nettoschuldenfrei (Barmittel >= Schulden): {nettofrei}')
    print(f'EBITDA <= 0 bei Nettoschulden: {ebitda_neg} (zählt als nicht erfüllt)')
    print(f'Nicht messbar (Schulden/Cash/EBITDA fehlen): {nicht_messbar}')
    print('\nSchwelle | erfüllt | Anteil')
    for t in THRESHOLDS_ND_EBITDA:
        passed = int(((base['Nettoschuldenfrei'] == True) | (ratio <= t)).sum())    # noqa: E712
        print(f'  <= {t:<4} | {passed:>7} | {passed / max(len(base), 1) * 100:5.1f} %')

    print(f'\n{line}\n5) CURRENT RATIO (Nicht-Finanzwerte, n={len(base)})\n{line}')
    cr = base['Current_Ratio']
    print('Verteilung:', _percentiles(cr))
    for t in (1.0, 1.5, 2.0):
        passed = int((cr >= t).sum())
        print(f'  Current Ratio >= {t}: {passed} von {len(base)} ({passed / max(len(base), 1) * 100:.1f} %)')
    print(f'  Nicht messbar: {int(cr.isna().sum())}')

    print(f'\n{line}\n6) GEWINNSTABILITÄT (Kriterium 4, alle Ticker mit Daten, n={int(df["Daten_ok"].sum())})\n{line}')
    ok = df[df['Daten_ok']]
    print('Anzahl Gewinnjahre je Ticker:', ok['Gewinnjahre'].value_counts().sort_index().to_dict())
    print(f'Gewinne in allen verfügbaren Jahren positiv: {int((ok["Gewinne_alle_positiv"] == True).sum())}')      # noqa: E712
    print(f'Dividende (Yield > 0): {int(ok["Dividende"].sum())} von {len(ok)}')

    print(f'\n{line}\n7) GEWINNWACHSTUM (Kriterium 6, NEU: Regressionssteigung / Durchschnittsgewinn)\n{line}')
    bewertbar = ok[ok['K6_bewertbar'] == True]                                                                  # noqa: E712
    print(f'Bewertbar (>= {K6_MIN_JAHRE} Jahre UND alle positiv): {len(bewertbar)} von {len(ok)}')
    nicht_bewertbar = ok[ok['K6_bewertbar'] != True]                                                            # noqa: E712
    if not nicht_bewertbar.empty:
        gruende = []
        for _, r in nicht_bewertbar.iterrows():
            if r['Gewinnjahre'] < K6_MIN_JAHRE:
                gruende.append(f"{r['Ticker']} (nur {r['Gewinnjahre']} Jahre)")
            elif r['Gewinne_alle_positiv'] is not True:
                gruende.append(f"{r['Ticker']} (Verlustjahr vorhanden)")
            else:
                gruende.append(f"{r['Ticker']} (unbekannt)")
        print('Nicht bewertbar:', ', '.join(gruende))
    kennzahl = bewertbar['K6_Kennzahl']
    print('\nVerteilung der Kennzahl (Steigung/Durchschnitt, entspricht ~Wachstum pro Jahr in % vom Gewinnniveau):')
    print(_percentiles(kennzahl))
    print('\nSchwelle | erfüllt | Anteil (bezogen auf bewertbare Ticker)')
    for t in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.33]:
        passed = int((kennzahl >= t).sum())
        print(f'  >= {t:<5} | {passed:>3} | {passed / max(len(bewertbar), 1) * 100:5.1f} %')
    print('\nSortiert nach Kennzahl:')
    print(bewertbar[['Ticker', 'Sektor', 'K6_Steigung_pro_Jahr', 'K6_Durchschnittsgewinn', 'K6_Kennzahl']]
          .sort_values('K6_Kennzahl', ascending=False).to_string(index=False))

    print(f'\n{line}\n8) EIGENKAPITALQUOTE (Ersatzkriterium Finanzwerte)\n{line}')
    fin_df = df[df['Finanzwert']]
    if fin_df.empty:
        print('Keine Finanzwerte nach der Erkennungsregel gefunden.')
    else:
        print(fin_df[['Ticker', 'Branche', 'Eigenkapitalquote_%']].to_string(index=False))
        print('Verteilung:', _percentiles(fin_df['Eigenkapitalquote_%']))
    print('\nZum Vergleich, Eigenkapitalquote der Nicht-Finanzwerte:', _percentiles(base['Eigenkapitalquote_%']))

    print(f'\n{line}\n9) ÜBERSICHT JE TICKER\n{line}')
    cols = ['Ticker', 'Sektor', 'Finanzwert', 'Current_Ratio', 'NettoSchulden_EBITDA',
            'Eigenkapitalquote_%', 'Gewinnjahre', 'K6_Kennzahl', 'Dividende', 'Status']
    print(df[cols].to_string(index=False))


def main() -> None:
    rows: List[Dict[str, Any]] = []
    failed: List[str] = []
    label_counter: Dict[str, Counter] = {k: Counter() for k in list(BS_LABELS) + list(FIN_LABELS)}

    for i, ticker in enumerate(SAMPLE, start=1):
        logger.info(f'[{i}/{len(SAMPLE)}] Verarbeite: {ticker}')
        try:
            info, financials, balance_sheet = fetch(ticker)
            row, labels_used = compute_metrics(ticker, info, financials, balance_sheet)
        except Exception as e:
            logger.error(f'{ticker}: {e}')
            failed.append(ticker)
            continue
        rows.append(row)
        if row['Daten_ok']:
            for key, label in labels_used.items():
                if label:
                    label_counter[key][label] += 1
        time.sleep(PAUSE_SECONDS)

    if not rows:
        logger.critical('Keine Daten abgerufen, Abbruch.')
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    logger.info(f'Ergebnis gespeichert: {OUTPUT_CSV}')
    print_summary(df, label_counter, len(SAMPLE), failed)


if __name__ == '__main__':
    main()
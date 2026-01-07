import os
import json
import yfinance as yf
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# --- KONFIGURATION ---
# Das Spreadsheet-ID aus der URL deiner Google Tabelle

SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

def get_graham_score(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        name = info.get('longName', ticker_symbol)
        current_price = info.get('currentPrice', 0)
        
        # Finanzdaten laden
        financials = stock.financials
        bs = stock.balance_sheet
        cf = stock.cashflow
        
        if financials.empty or bs.empty:
            return None

        # --- TEIL A: GRAHAM SCORE (0-7) ---
        graham_score = 0
        mcap = info.get('marketCap', 0)
        if mcap >= 2_000_000_000: graham_score += 1 # 1. Größe
        
        latest_bs = bs.iloc[:, 0]
        ca = latest_bs.get('Total Current Assets', 0)
        cl = latest_bs.get('Total Current Liabilities', 0)
        if cl > 0 and (ca / cl) >= 2.0: graham_score += 1 # 2. Liquidität
        
        debt = latest_bs.get('Total Debt', 0)
        if debt < (ca - cl): graham_score += 1 # 3. Schulden vs. Working Cap
        
        if 'Net Income' in financials.index:
            net_inc = financials.loc['Net Income']
            if (net_inc > 0).all(): graham_score += 1 # 4. Stabilität
            if len(net_inc) >= 3 and net_inc.iloc[0] > net_inc.iloc[-1]: graham_score += 1 # 6. Wachstum
            
        if info.get('dividendYield', 0) > 0: graham_score += 1 # 5. Dividende
        
        pe = info.get('trailingPE') or 0
        pb = info.get('priceToBook') or 0
        if 0 < pe <= 15 and 0 < pb <= 1.5 and (pe * pb) <= 22.5: graham_score += 1 # 7. Multiplier

        # --- TEIL B: BONUS QUALITY SCORE (0-3) ---
        bonus_score = 0
        # 1. Aktienrückkäufe (Sinkende Anzahl Aktien)
        if len(bs.columns) > 1:
            shares_curr = bs.iloc[:, 0].get('Ordinary Share Number', 0)
            shares_prev = bs.iloc[:, 1].get('Ordinary Share Number', 0)
            if shares_curr > 0 and shares_prev > 0 and shares_curr < shares_prev:
                bonus_score += 1
        
        # 2. Positiver Free Cashflow
        if not cf.empty and 'Free Cash Flow' in cf.index:
            if cf.loc['Free Cash Flow'].iloc[0] > 0:
                bonus_score += 1
        
        # 3. Kapitalrendite (ROIC > 15%)
        equity = latest_bs.get('Stockholders Equity', 1)
        invested_capital = equity + debt
        if 'Net Income' in financials.index and invested_capital > 0:
            roic = financials.loc['Net Income'].iloc[0] / invested_capital
            if roic > 0.15:
                bonus_score += 1

        # --- TEIL C: PREISE ---
        eps = info.get('trailingEps', 0)
        book_value = info.get('bookValue', 0)
        if eps > 0 and book_value > 0:
            graham_price = (22.5 * eps * book_value) ** 0.5
        else:
            graham_price = 0
            
        margin_of_safety = 0
        if graham_price > 0:
            margin_of_safety = (1 - (current_price / graham_price)) * 100

        # --- EXPORT DATENSATZ (10 Spalten) ---
        return [
            datetime.now().strftime("%d.%m.%Y %H:%M"), # A
            ticker_symbol,                             # B
            name,                                      # C
            graham_score,                              # D
            bonus_score,                               # E (NEU)
            round(current_price, 2),                   # F
            round(graham_price, 2),                    # G
            f"{round(margin_of_safety, 1)}%",          # H
            round(pe, 2),                              # I
            round(mcap / 1e9, 2)                       # J
        ]

    except Exception as e:
        print(f"Fehler bei {ticker_symbol}: {e}")
        return None
        
def main():
    # 1. Authentifizierung über GitHub Secret
    creds_json = os.getenv('GOOGLE_CREDS_JSON')
    if not creds_json:
        print("Fehler: GOOGLE_CREDS_JSON nicht gefunden.")
        return

    creds_dict = json.loads(creds_json)
    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    
    # 2. Tabelle öffnen
    sheet = gc.open_by_key(SPREADSHEET_ID)
    input_ws = sheet.worksheet('stocklist')
    output_ws = sheet.worksheet('Buy-triggers')
    
    tickers = [t for t in input_ws.col_values(1)[1:] if t.strip()]
    
    results = []
    for symbol in tickers:
        if not symbol: continue
        print(f"Verarbeite: {symbol}")
        data = get_graham_score(symbol.strip())
        if data:
            results.append(data)
    
    # 4. Ergebnisse gesammelt schreiben
    if results:
        output_ws.append_rows(results, value_input_option='USER_ENTERED')
        print(f"Erfolgreich {len(results)} Zeilen hinzugefügt.")

if __name__ == "__main__":
    main()

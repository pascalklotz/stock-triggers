import os
import json
import yfinance as yf
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# --- KONFIGURATION ---
# Das Spreadsheet-ID aus der URL deiner Google Tabelle
SPREADSHEET_ID = '1k0qv72oN2V6E4NCwYTFjtvcGHNrmiFlczlqa66ke47I'

def get_graham_score(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Basis-Daten
        name = info.get('longName', ticker_symbol)
        mcap = info.get('marketCap', 0)
        pe = info.get('trailingPE') or info.get('forwardPE') or 0
        pb = info.get('priceToBook') or 0
        
        # Score Berechnung (Basis 5 Punkte)
        score = 0
        if mcap >= 2_000_000_000: score += 1 # Größe
        if 0 < pe <= 15: score += 1           # P/E Ratio
        if 0 < (pe * pb) <= 22.5: score += 1  # Graham Multiplier
        
        # Finanzen für Current Ratio & Debt/Equity
        # Wir nutzen fast_info oder financials (yfinance ist hier manchmal volatil)
        try:
            bs = stock.balance_sheet
            if not bs.empty:
                # Suche nach Labels (können variieren)
                ca = bs.iloc[:, 0].get('Current Assets', 0)
                cl = bs.iloc[:, 0].get('Current Liabilities', 0)
                debt = bs.iloc[:, 0].get('Total Debt', 0)
                equity = bs.iloc[:, 0].get('Stockholders Equity', 1)
                
                if cl > 0 and (ca / cl) >= 2.0: score += 1
                if (debt / equity) <= 1.0: score += 1
        except:
            pass # Wenn Bilanzdaten fehlen, bleibt der Score niedriger
            
        return [
            datetime.now().strftime("%d.%m.%Y %H:%M"),
            ticker_symbol,
            name,
            score,
            round(pe, 2),
            round(mcap / 1e9, 2)
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

    # 3. Ticker holen (Spalte A ab Zeile 2, max 200)
    tickers = input_ws.col_values(1)[1:201]
    
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

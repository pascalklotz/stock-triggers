import os
import json
import yfinance as yf
import pandas as pd
import gspread
import numpy as np
from google.oauth2.service_account import Credentials
from datetime import datetime

# --- KONFIGURATION ---
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

class SellScore:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.sell_score = 0
        self.evaluations = {}
        self.company_name = "N/A"
        try:
            self.info = self.stock.info
            self.balance_sheet = self.stock.balance_sheet
            self.income_statement = self.stock.financials
            self.company_name = self.info.get('longName', 'N/A')
            self.latest_bs = self.balance_sheet.iloc[:, 0] if not self.balance_sheet.empty else pd.Series()
            self.data_available = not self.balance_sheet.empty
        except:
            self.data_available = False

    def check_sell_criteria(self):
        if not self.data_available: return
        
        # K1: Liquidität
        ca = self.latest_bs.get('Total Current Assets', 0)
        cl = self.latest_bs.get('Total Current Liabilities', 1)
        curr_ratio = ca / cl
        if curr_ratio < 1.5: self.sell_score += 1

        # K2: Verschuldung
        debt = self.latest_bs.get('Total Debt', 0)
        equity = self.latest_bs.get('Stockholders Equity', 1)
        if equity <= 0 or (debt / equity) > 2.0: self.sell_score += 1

        # K3/K4: Gewinne
        try:
            ni = self.income_statement.loc['Net Income']
            if (ni <= 0).sum() >= 2: self.sell_score += 1 # K3: Verlustjahre
            if ni.iloc[0] < ni.iloc[-1]: self.sell_score += 1 # K4: Negatives Wachstum
        except: pass

        # K5: Überbewertung
        pe = self.info.get('trailingPE', 0)
        pb = self.info.get('priceToBook', 0)
        if pe * pb > 45: self.sell_score += 1

    def get_row(self):
        self.check_sell_criteria()
        return [
            datetime.now().strftime("%d.%m.%Y %H:%M"),
            self.ticker,
            self.company_name,
            self.sell_score,
            "🛑 VERKAUFEN" if self.sell_score >= 3 else "✅ HALTEN"
        ]

def main():
    # Auth
    creds_json = os.getenv('GOOGLE_CREDS_JSON')
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(creds)
    
    sheet = gc.open_by_key(SPREADSHEET_ID)
    
    # Ticker aus 'portfolio' lesen (Spalte A)
    portfolio_ws = sheet.worksheet('portfolio')
    tickers = [t for t in portfolio_ws.col_values(1)[1:] if t.strip()]
    
    results = []
    for t in tickers:
        print(f"Check Portfolio: {t}")
        evaluator = SellScore(t)
        results.append(evaluator.get_row())
    
    # In 'sell-triggers' schreiben
    if results:
        output_ws = sheet.worksheet('sell-triggers')
        output_ws.append_rows(results, value_input_option='USER_ENTERED')

if __name__ == "__main__":
    main()

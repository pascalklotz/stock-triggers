import os
import json
import yfinance as yf
import pandas as pd
import gspread
import numpy as np
from google.oauth2.service_account import Credentials
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- KONFIGURATION ---
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')


class SellScore:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.sell_score = 0
        self.evaluations = {}
        self.company_name = "N/A"
        self.metrics = {}
        try:
            self.info = self.stock.info
            self.balance_sheet = self.stock.balance_sheet
            self.income_statement = self.stock.financials
            self.company_name = self.info.get('longName', 'N/A')
            self.latest_bs = self.balance_sheet.iloc[:, 0] if not self.balance_sheet.empty else pd.Series()
            self.data_available = not self.balance_sheet.empty
        except:
            self.data_available = False

    def calculate_metrics(self):
        """Berechnet alle Metriken einmalig und speichert sie."""
        if not self.data_available: return
        
        # 1. Liquidität
        ca = (self.latest_bs.get('Total Current Assets') or 
              self.latest_bs.get('Current Assets') or 0)
        cl = (self.latest_bs.get('Total Current Liabilities') or 
              self.latest_bs.get('Current Liabilities') or 1) # Vermeidung Division durch 0
        
        self.metrics['curr_ratio'] = round(ca / cl, 2)

        # 2. Verschuldung
        debt = self.latest_bs.get('Total Debt', 0)
        equity = self.latest_bs.get('Stockholders Equity', 1)
        self.metrics['debt_equity'] = round(debt / equity, 2) if equity > 0 else float('inf')

        # 3. Gewinne
        self.metrics['neg_years'] = 0
        self.metrics['declining_growth'] = False
        try:
            if 'Net Income' in self.income_statement.index:
                ni = self.income_statement.loc['Net Income']
                self.metrics['neg_years'] = (ni <= 0).sum()
                # iloc[0] ist aktuell, iloc[-1] ist alt. Wenn aktuell < alt, dann Rückgang.
                if len(ni) > 1 and ni.iloc[0] < ni.iloc[-1]:
                    self.metrics['declining_growth'] = True
        except: pass

        # 4. Bewertung
        pe = self.info.get('trailingPE', 0)
        pb = self.info.get('priceToBook', 0)
        self.metrics['graham_mult'] = round(pe * pb, 1) if pe and pb else 0
        self.metrics['pe'] = round(pe, 1) if pe else 0

    def check_sell_criteria(self):
        if not self.data_available: return
        
        self.sell_score = 0 # Reset Score
        
        if self.metrics.get('curr_ratio', 0) < 1.5: self.sell_score += 1
        if self.metrics.get('debt_equity', 0) > 2.0: self.sell_score += 1
        if self.metrics.get('neg_years', 0) >= 2: self.sell_score += 1
        if self.metrics.get('declining_growth', False): self.sell_score += 1
        if self.metrics.get('graham_mult', 0) > 45: self.sell_score += 1

    def get_row(self):
        self.calculate_metrics()
        self.check_sell_criteria()
        
        # Info Text aus berechneten Metriken bauen
        m = self.metrics
        ni_info = f"{m.get('neg_years', 0)} J. neg."
        info_text = f"CR: {m.get('curr_ratio')} | D/E: {m.get('debt_equity')} | Mult: {m.get('graham_mult')} | {ni_info} | P/E: {m.get('pe')}"

        # Rückgabe der Spalten für Google Sheets
        return [
            datetime.now().strftime("%d.%m.%Y %H:%M"), # A: Datum
            self.ticker,                             # B: Ticker
            self.company_name,                       # C: Name
            self.sell_score,                         # D: Score
            "🛑 VERKAUFEN" if self.sell_score >= 3 else "✅ HALTEN", # E: Status
            info_text                                # F: Die neue Info-Spalte
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

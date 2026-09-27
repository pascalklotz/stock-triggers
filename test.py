import yfinance as yf
print(yf.Ticker("GNTX").balance_sheet.index.tolist())
for t in ["ALLE", "ICE"]:
    print(t, yf.Ticker(t).financials.loc["Net Income"].to_dict())
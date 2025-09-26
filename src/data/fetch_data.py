import yfinance as yf
import pandas as pd
import os
from datetime import date

root_dir = os.getcwd()   # Gets current working directory
#print("Root directory:", root_dir)
DATA_DIR = os.path.join(root_dir, "data", "raw")

def fetch_stock(symbol, start, end, data_dir):
    end = end or date.today().isoformat()
    ticker = f"{symbol.upper()}.NS"
    filename = data_dir / f"{symbol.upper()}_{end}.csv"
    if filename.exists():
        return filename
    data = yf.download(ticker, start=start, end=end)
    filename.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filename)
    return filename

def fetch_stock1(symbol="INFY", start="2020-01-01", end="2025-01-01"):
    # Ensure NSE symbol ends with .NS
    ticker = f"{symbol.upper()}.NS"
    data = yf.download(ticker, start=start, end=end)
    
    # reset so Date is a normal column
    data.reset_index(inplace=True)

    # drop multi-index headers if any (ensure only one header row)
    data = pd.DataFrame(data)

    ##print("File path " +DATA_DIR)
    ##print("File path " +os.path.join(DATA_DIR, f"{symbol}.csv"))


    # Save CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    # data.to_csv(f"../../data/raw/{symbol}.csv")
    # #print(f"Saved {symbol} data to CSV")
    save_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    data.to_csv(save_path, index=False)   # no extra index junk
    #print(f"✅ Saved {symbol} to {save_path}")

    return data

if __name__ == "__main__":
    df = fetch_stock("INFY")
    #print(df.head())

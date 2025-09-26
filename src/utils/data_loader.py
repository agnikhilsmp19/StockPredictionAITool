import os
import pandas as pd
from pathlib import Path

# Root directory of your project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")

def load_and_clean_stock_data(ticker: str) -> pd.DataFrame:
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # Drop any rows that are obviously headers or invalid
    data = data.dropna(how="any")  # remove NaN
    data = data[~data["Close"].astype(str).str.contains("[A-Za-z]")]  # drop rows with text

    # Ensure all numeric columns are floats
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop again after coercion in case of NaN
    data = data.dropna()

    # Clean OHLCV columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows with NaN after coercion (these were strings like 'INFY.NS')
    data = data.dropna(subset=["Close", "Open", "High", "Low", "Volume"])

    #print(data.head(), data.dtypes)  # sanity check
    # Ensure only Close column
    data = data.reset_index()[['Date', 'Close']]

    return data

def load_and_clean_stock_data_lstm(ticker, data_dir=None, date_str=None):
    # Build path dynamically based on ticker and date (today by default)
    from datetime import date
    if date_str is None:
        date_str = date.today().isoformat()
    if data_dir is None:
        ROOT = Path(__file__).resolve().parents[2]
        data_dir = ROOT / "data" / "raw"
    fname = data_dir / f"{ticker.upper()}_{date_str}.csv"
    if not fname.exists():
        raise FileNotFoundError(f"{fname} not found. Please fetch latest CSV for {ticker}.")
    data = pd.read_csv(fname, index_col=0, parse_dates=True)
    # Make sure Close is float and drop missing
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    
    data = data.dropna(subset=["Close"])

    data["High"] = pd.to_numeric(data["High"], errors="coerce")
    data["Low"] = pd.to_numeric(data["Low"], errors="coerce")
    data["Open"] = pd.to_numeric(data["Open"], errors="coerce")
    return data
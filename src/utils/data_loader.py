import os
import pandas as pd

# Root directory of your project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")

def load_and_clean_stock_data(ticker: str) -> pd.DataFrame:
    """
    Load stock data from CSV and clean it:
    - Removes invalid rows like 'INFY.NS' headers in data
    - Converts OHLCV columns to numeric
    - Ensures Date is parsed as datetime index (if present)
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV not found for {ticker}: {file_path}")

    # Load CSV
    data = pd.read_csv(file_path)

    # If "Date" exists, make it datetime index
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.set_index("Date")

    # Drop invalid rows (like strings in Close column)
    data = data[pd.to_numeric(data["Close"], errors="coerce").notnull()]

    # Convert OHLCV to numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data

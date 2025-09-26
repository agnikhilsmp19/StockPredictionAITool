import os
import pandas as pd

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

    print(data.head(), data.dtypes)  # sanity check
    # Ensure only Close column
    data = data.reset_index()[['Date', 'Close']]

    return data

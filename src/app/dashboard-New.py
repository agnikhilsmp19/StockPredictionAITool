import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
import ta

# --- Root Paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PRED = ROOT / "data" / "predictions"

# --- Secure Access ---
ALLOWED_EMAILS = ["agnikhil82@gmail.com","agnikhilsmp19@gmail.com"]
user_email = st.text_input("Enter your email to access")
if user_email not in ALLOWED_EMAILS:
    st.warning("Access denied. Contact admin.")
    st.stop()

# --- Streamlit UI Inputs ---
st.title("Indian Stock Prediction Agentic AI")
ticker = st.text_input("Enter NSE Stock Ticker", "INFY")
model_type = st.radio("Select Prediction Model", ["Linear Regression", "LSTM"])
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

raw_file = DATA_RAW / f"{ticker.upper()}.csv"
if not raw_file.exists():
    st.warning("Stock data not found. Please fetch the stock data first.")
    st.stop()
data = pd.read_csv(raw_file, index_col=0, parse_dates=True)
# Force numeric columns just in case
for col in ["Open", "High", "Low", "Close", "Volume"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# --- Technical Indicator Calculation ---
data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# --- Display Data ---
st.write(f"### {ticker.upper()} Data Preview", data.tail())
st.write("### Closing Price")
st.line_chart(data["Close"])
st.write("### SMA20 (20-Day Simple Moving Average)")
st.line_chart(data[['Close', 'SMA20']].dropna())
st.write("### RSI (14-Day Relative Strength Index)")
st.line_chart(data['RSI'].dropna())

# --- Prediction Section ---
def decide_action(current, predicted):
    if predicted > current * 1.01:
        return "Buy"
    elif predicted < current * 0.99:
        return "Sell"
    else:
        return "Hold"

if model_type == "Linear Regression":
    # ---- Linear Regression Model ----
    data["Target"] = data["Close"].shift(-1)
    data.dropna(inplace=True)
    X = data[["Close"]]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    st.write(f"**Prediction RMSE:** {rmse:.2f}")

    pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}, index=y_test.index)
    st.write("### Prediction vs Actual (Linear Regression)")
    st.line_chart(pred_df)
    data_test = X_test.copy()
    data_test["Predicted"] = y_pred
    data_test["Decision"] = data_test.apply(lambda row: decide_action(row["Close"], row["Predicted"]), axis=1)

elif model_type == "LSTM":
    # ---- LSTM Model: Load Precomputed Predictions ----
    pred_file = DATA_PRED / f"lstm_pred_{ticker.upper()}.csv"
    if pred_file.exists():
        lstm_pred = pd.read_csv(pred_file, index_col=0, parse_dates=True)
        st.write("### LSTM Model Predictions")
        st.line_chart(lstm_pred)
        # Optionally, run decision logic using last values from 'Close' and 'Predicted'
        lstm_pred["Decision"] = [decide_action(c, p) for c, p in zip(lstm_pred["Actual"], lstm_pred["Predicted"])]
        st.write("### Agent Decisions (LSTM)", lstm_pred.tail())
        data_test = lstm_pred[["Actual", "Predicted", "Decision"]].copy()
        # For a simulation, you can adapt the backtest logic below to predict columns
    else:
        st.warning("LSTM predictions not found for this ticker. Run LSTM model script first.")
        st.stop()

# --- Agent-Based Backtesting Simulation ---
cash = 100000
stock = 0
trades = []
for i, row in data_test.iterrows():
    current = row["Close"] if "Close" in row else row["Actual"]
    predicted = row["Predicted"]
    decision = row["Decision"]
    if decision == "Buy" and cash > 0:
        stock = cash / current
        cash = 0
        trades.append(f"Day {i}: Buy at {current:.2f}")
    elif decision == "Sell" and stock > 0:
        cash = stock * current
        stock = 0
        trades.append(f"Day {i}: Sell at {current:.2f}")
portfolio_value = cash + (stock * (data_test["Close"].iloc[-1] if "Close" in data_test else data_test["Actual"].iloc[-1]))
st.write(f"### Backtesting Simulation\n**Final Portfolio Value:** ₹{portfolio_value:,.2f}")
st.write("Sample Trades:")
for t in trades[-5:]:
    st.write(t)

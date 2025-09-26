import streamlit as st
import pandas as pd
import ta
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datetime import date
from pathlib import Path

from src.utils.paths import DATA_RAW, DATA_PRED
from src.data.fetch_data import fetch_stock
from src.agent.decision_engine import decide_action, decision_icon
from src.agent.backtest import backtest   # ✅ enabled backtest
from src.models.linear_model import run_linear_regression




# --- Sidebar UI ---
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("NSE Stock Symbol", "INFY")
model_type = st.sidebar.radio("Prediction Model", ["Linear Regression", "LSTM"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = date.today().isoformat()

user_email = st.sidebar.text_input("Enter your email to access")
ALLOWED_EMAILS = ["test@test.com"]
if user_email not in ALLOWED_EMAILS:
    st.warning("Access denied. Contact admin.")
    st.stop()

st.title("📊 Indian Stock Prediction Agentic AI Dashboard")

# --- Fetch or Load Stock Data ---
stock_file = DATA_RAW / f"{ticker.upper()}_{end_date}.csv"
refresh_clicked = st.sidebar.button("🔄 Refresh Stock Data")

if refresh_clicked or not stock_file.exists():
    with st.spinner("Fetching latest stock data..."):
        filename = fetch_stock(
            symbol=ticker,
            start=start_date.isoformat(),
            end=end_date,
            data_dir=DATA_RAW,
        )
        st.sidebar.success(f"{ticker.upper()} data refreshed ({filename.name})")
    stock_file = DATA_RAW / f"{ticker.upper()}_{end_date}.csv"

if not stock_file.exists():
    st.warning(f"No data file for today ({end_date}). Please refresh!")
    st.stop()
else:
    st.info(f"Using data file: {stock_file.name}")

# --- Load Data ---
data = pd.read_csv(stock_file, index_col=0, parse_dates=True)

# ✅ Ensure numeric Close column
if "Close" in data.columns:
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Close"])
else:
    st.error("⚠️ No 'Close' column found in stock file.")
    st.stop()

st.write(f"#### Latest {ticker.upper()} Price Data")
st.dataframe(data.tail())

# --- Charts & Indicators ---
st.write("#### 📈 Closing Price Trend")
st.line_chart(data["Close"])

# Add technical indicators
data["SMA20"] = ta.trend.sma_indicator(data["Close"], window=20)
data["RSI"] = ta.momentum.rsi(data["Close"], window=14)

st.write("#### 🟦 SMA20 (20-Day Simple Moving Average)")
st.line_chart(data[["Close", "SMA20"]].dropna())

st.write("#### 🟩 RSI (14-Day Relative Strength Index)")
st.line_chart(data["RSI"].dropna())

# --- Prediction Models ---
if model_type == "Linear Regression":
    rmse, pred_df, data_test = run_linear_regression(data)
    st.write(f"RMSE: **{rmse:.2f}**")

    st.write("#### Prediction vs Actual (Linear Regression)")
    st.line_chart(pred_df)

    data_test["Decision"] = data_test.apply(
        lambda row: decide_action(row["Close"], row["Predicted"]), axis=1
    )
    data_test["Icon"] = data_test["Decision"].map(decision_icon)

elif model_type == "LSTM":
    pred_file = DATA_PRED / f"lstm_pred_{ticker.upper()}.csv"
    if pred_file.exists():
        lstm_pred = pd.read_csv(pred_file, index_col=0, parse_dates=True)
        st.write("#### Prediction vs Actual (LSTM)")
        st.line_chart(lstm_pred[["Actual", "Predicted"]])

        lstm_pred["Decision"] = [
            decide_action(c, p) for c, p in zip(lstm_pred["Actual"], lstm_pred["Predicted"])
        ]
        lstm_pred["Icon"] = lstm_pred["Decision"].map(decision_icon)

        st.write("#### Recent Agent Decisions (LSTM)")
        st.dataframe(lstm_pred.tail()[["Decision", "Icon"]])

        data_test = lstm_pred.copy()
    else:
        st.warning("LSTM predictions not found. Please run and save output into /data/predictions.")
        st.stop()

# --- Agent Decisions ---
st.write("#### 🧠 Recent Agent Decisions")
st.dataframe(data_test.tail()[["Decision", "Icon"]])

# --- Backtest ---
final_val, last_trades = backtest(data_test)
st.write(f"#### 🧾 Portfolio Backtest Value: ₹{final_val:,.2f}")
st.write("Last 5 trades:")
for t in last_trades:
    st.write(t)

# --- Sidebar Instructions ---
st.sidebar.markdown(
    """
**Instructions:**  
- Enter any Indian NSE stock  
- Choose prediction model  
- Click **Refresh**  
- View predictions and agent decisions below.  
"""
)

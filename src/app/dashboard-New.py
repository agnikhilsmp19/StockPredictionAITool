import streamlit as st
import pandas as pd
import ta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from datetime import date
from pathlib import Path

from src.utils.paths import DATA_RAW, DATA_PRED
from src.data.fetch_data import fetch_stock
from src.agent.decision_engine import decide_action, decision_icon
from src.agent.backtest import backtest  # Ensure your updated backtest can take price_column
from src.models.linear_model import run_linear_regression

# --- Sidebar UI with Form for Mobile Friendliness ---
with st.sidebar.form(key="user-input-form"):
    ticker = st.text_input("NSE Stock Symbol", "INFY")
    model_type = st.radio("Prediction Model", ["Linear Regression", "LSTM"])
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    user_email = st.text_input("Enter your email to access")
    submitted = st.form_submit_button("Submit")

# Persist inputs in session state for reliable behavior
if submitted:
    st.session_state['ticker'] = ticker.upper()
    st.session_state['model_type'] = model_type
    st.session_state['start_date'] = start_date
    st.session_state['user_email'] = user_email

def get_ses(key, default=None):
    return st.session_state.get(key, default)

ticker = get_ses('ticker', 'INFY')
model_type = get_ses('model_type', 'Linear Regression')
start_date = get_ses('start_date', pd.to_datetime("2020-01-01"))
user_email = get_ses('user_email', '')

end_date = date.today().isoformat()
ALLOWED_EMAILS = ["test@test.com"]
if user_email.strip().lower() not in [email.lower() for email in ALLOWED_EMAILS]:
    st.warning("Access denied. Contact admin.")
    st.stop()

st.title("📊 Indian Stock Prediction Agentic AI Dashboard")

# --- Fetch or Load Stock Data (with Refresh Button below form) ---
stock_file = DATA_RAW / f"{ticker}_{end_date}.csv"
refresh_clicked = st.sidebar.button("🔄 Refresh Stock Data")

if refresh_clicked or not stock_file.exists():
    with st.spinner("Fetching latest stock data..."):
        filename = fetch_stock(
            symbol=ticker,
            start=start_date.isoformat(),
            end=end_date,
            data_dir=DATA_RAW,
        )
        st.sidebar.success(f"{ticker} data refreshed ({filename.name})")
    stock_file = DATA_RAW / f"{ticker}_{end_date}.csv"

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

st.write(f"#### Latest {ticker} Price Data")
st.dataframe(data.tail())

# --- Charts & Indicators ---
st.write("#### 📈 Closing Price Trend")
st.line_chart(data["Close"])
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
    pred_file = DATA_PRED / f"lstm_pred_{ticker}.csv"
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

# Determine column for backtest
price_col = "Close" if "Close" in data_test.columns else "Actual"
final_val, last_trades = backtest(data_test, price_column=price_col)

st.write(f"#### 🧾 Portfolio Backtest Value: ₹{final_val:,.2f}")
st.write("Last 5 trades:")
for t in last_trades[-5:]:
    st.write(t)

# --- Sidebar Instructions ---
st.sidebar.markdown(
    """
**Instructions:**  
- Enter any Indian NSE stock  
- Choose prediction model  
- Enter your email  
- Press **Submit/Update**  
- Optionally click **Refresh Stock Data** if outdated  
- View predictions and agent decisions below.  
"""
)

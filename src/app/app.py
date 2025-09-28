import streamlit as st
import pandas as pd
import ta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datetime import date
from pathlib import Path

# Your helper/module imports...
from src.utils.paths import DATA_RAW, DATA_PRED
from src.data.fetch_data import fetch_stock
from src.agent.decision_engine import decide_action, decision_icon
from src.agent.backtest import backtest
from src.models.linear_model import run_linear_regression

ALLOWED_EMAILS = ["test@test.com", "agnikhilsmp19@gmail.com"]

def is_authenticated(email):
    return email and email.strip().lower() in [addr.lower() for addr in ALLOWED_EMAILS]

def login_screen():
    st.header("🛡️ Login to Stock Prediction Agentic AI")
    email = st.text_input("Enter your email to access")
    login = st.button("Login")
    if login:
        if is_authenticated(email):
            st.session_state["is_logged_in"] = True
            st.session_state["user_email"] = email
            st.success("Login successful! Loading dashboard...")
            #st.experimental_rerun()
            st.rerun()
        else:
            st.error("Access denied. Please contact admin.")
            st.stop()
    st.stop()

def dashboard():
    st.title("📊 Indian Stock Prediction Agentic AI Dashboard")
    logout = st.button("Logout")
    if logout:
        st.session_state.clear()
        st.success("You have logged out.")
        #st.experimental_rerun()
        st.rerun()
    # --- ALL DASHBOARD CONTROLS: in main area ---
    with st.form("controls-form"):
        ticker = st.text_input("NSE Stock Symbol", st.session_state.get('ticker', 'INFY'))
        model_type = st.radio("Prediction Model", ["Linear Regression", "LSTM"], index=0)
        start_date = st.date_input("Start Date", st.session_state.get('start_date', pd.to_datetime("2020-01-01")))
        controls_submitted = st.form_submit_button("Update")
    if controls_submitted:
        st.session_state['ticker'] = ticker.upper()
        st.session_state['model_type'] = model_type
        st.session_state['start_date'] = start_date
    ticker = st.session_state.get('ticker', 'INFY')
    model_type = st.session_state.get('model_type', "Linear Regression")
    start_date = st.session_state.get('start_date', pd.to_datetime("2020-01-01"))
    end_date = date.today().isoformat()
    refresh_clicked = st.button("🔄 Refresh Stock Data")
    stock_file = DATA_RAW / f"{ticker}_{end_date}.csv"
    print("start_date - app.py",start_date)
    #print("start_date - app.py",start_date.date())
    if refresh_clicked or not stock_file.exists():
        with st.spinner("Fetching/updating stock data..."):
            fetch_stock(
                symbol=ticker,
                start=start_date.isoformat(),
                end=end_date,
                data_dir=DATA_RAW,
            )
            st.success(f"{ticker} data refreshed.")
        stock_file = DATA_RAW / f"{ticker}_{end_date}.csv"
    if not stock_file.exists():
        st.warning(f"No data found for today ({end_date}). Please refresh using the button above!")
        st.stop()
    else:
        st.info(f"Using file: {stock_file.name}")
    data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    for col in ["Open", "High", "Low", "Close"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").round(2)
    data = data.dropna(subset=["Close"])
    st.write(f"#### Latest {ticker} Price Data")
    st.dataframe(data.tail())
    st.write("#### 📈 Closing Price Trend")
    st.line_chart(data["Close"])
    data["SMA20"] = ta.trend.sma_indicator(data["Close"], window=20).round(2)
    data["RSI"] = ta.momentum.rsi(data["Close"], window=14).round(2)
    st.write("#### 🟦 SMA20 (20-Day Simple Moving Average)")
    st.line_chart(data[["Close", "SMA20"]].dropna())
    st.write("#### 🟩 RSI (14-Day Relative Strength Index)")
    st.line_chart(data["RSI"].dropna())
    if model_type == "Linear Regression":
        rmse, pred_df, data_test = run_linear_regression(data)
        st.write(f"RMSE: **{rmse:.2f}**")
        st.write("#### Prediction vs Actual (Linear Regression)")
        pred_df = pred_df.round(2)
        st.line_chart(pred_df)
        data_test["Decision"] = data_test.apply(lambda row: decide_action(row["Close"], row["Predicted"]), axis=1)
        data_test["Icon"] = data_test["Decision"].map(decision_icon)
    elif model_type == "LSTM":
        pred_file = DATA_PRED / f"lstm_pred_{ticker}.csv"
        if pred_file.exists():
            lstm_pred = pd.read_csv(pred_file, index_col=0, parse_dates=True)
            lstm_pred[["Actual", "Predicted"]] = lstm_pred[["Actual", "Predicted"]].round(2)
            st.write("#### Prediction vs Actual (LSTM)")
            st.line_chart(lstm_pred[["Actual", "Predicted"]])
            lstm_pred["Decision"] = [decide_action(c, p) for c, p in zip(lstm_pred["Actual"], lstm_pred["Predicted"])]
            lstm_pred["Icon"] = lstm_pred["Decision"].map(decision_icon)
            st.write("#### Recent Agent Decisions (LSTM)")
            st.dataframe(lstm_pred.tail()[["Decision", "Icon"]])
            data_test = lstm_pred.copy()
        else:
            st.warning("LSTM predictions not found. Please run and save output into /data/predictions.")
            st.stop()
    st.write("#### 🧠 Recent Agent Decisions")
    st.dataframe(data_test.tail()[["Decision", "Icon"]])
    

    if "Decision" in data_test.columns:
        price_col = "Close" if "Close" in data_test.columns else "Actual"
        final_val, last_trades = backtest(data_test, price_column=price_col)

        st.write(f"#### 🧾 Portfolio Backtest Value: ₹{final_val:,.2f}")
        st.write("Last 5 trades:")
        for t in last_trades[-5:]:
            st.write(t)
    else:
        st.warning("⚠️ No 'Decision' column found in data. Backtest skipped.")

    
    st.markdown(
        """
    **Instructions:**  
    - Enter any Indian NSE stock symbol  
    - Choose prediction model  
    - Click Update  
    - Optionally click Refresh Data  
    - View predictions and agent decisions below  
    """
    )

# --- MAIN APP ROUTING ---
if not st.session_state.get("is_logged_in"):
    login_screen()
else:
    dashboard()

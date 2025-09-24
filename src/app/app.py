import sys
import os
import streamlit as st


sys.path.append(os.path.abspath("."))  # add project root to Python path

from src.agent.prediction_engine import predict_stock
from src.agent.decision_engine import make_decision
from src.agent.indicators import calculate_rsi

st.title("📊 Stock Prediction Agentic AI Tool (POC)")
ticker = st.text_input("Enter NSE Stock Ticker (e.g., INFY.NS)", "INFY.NS")
days = st.slider("Prediction Days", 1, 10, 5)

if st.button("Predict"):
    preds, df = predict_stock(ticker, days)
    final_decision, explanation = make_decision(preds, df)

    st.line_chart(df["Close"].tail(200))
    st.write(f"Predicted Prices: {preds}")

    st.subheader("🤖 Final AI Decision")
    if final_decision == "BUY":
        st.success("BUY 📈")
    elif final_decision == "SELL":
        st.error("SELL 📉")
    else:
        st.info("HOLD 🤝")

    with st.expander("See Decision Explanation"):
        for reason in explanation:
            st.write("- " + reason)

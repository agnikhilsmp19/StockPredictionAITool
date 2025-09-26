import streamlit as st
import pandas as pd
import os
import ta

st.title("Indian Stock Prediction POC")

# Sidebar - User input
ticker = st.text_input("Enter NSE Stock Ticker", "INFY")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

root_dir = os.getcwd()   # Gets current working directory
#print("Root directory:", root_dir)
DATA_DIR = os.path.join(root_dir, "data", "raw")

data_path = os.path.join(DATA_DIR, f"{ticker}.csv")
#print(data_path)
try:
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data = data[data["Close"] != "INFY.NS"]
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    df = data.dropna()

    #print(data.head())
    
    st.write(f"### {ticker} Data Preview")
    st.write(data.tail())
    st.line_chart(data['Close'])
except FileNotFoundError:
    st.warning("Stock data not found. Please fetch the stock data first using fetch_data.py")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data["Target"] = data["Close"].shift(-1)
data.dropna(inplace=True)

X = data[["Close"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("### Prediction vs Actual")
pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}, index=y_test.index)
st.line_chart(pred_df)

data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

st.write("### Closing Price with SMA20")
st.line_chart(data[['Close','SMA20']])

st.write("### RSI")
st.line_chart(data['RSI'])
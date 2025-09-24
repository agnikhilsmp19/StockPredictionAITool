import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

def predict_stock(ticker="INFY.NS", days=5):
    df = yf.download(ticker, period="5y")
    data = df["Close"].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(60, len(scaled_data)):
        x.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    x, y = np.array(x), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x.shape[1],1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)

    last_60 = scaled_data[-60:]
    preds = []
    curr = last_60.reshape(1,60,1)

    for _ in range(days):
        pred = model.predict(curr, verbose=0)  # shape (1,1)
        preds.append(scaler.inverse_transform(pred)[0][0])
        curr = np.append(curr[:,1:,:], pred.reshape(1,1,1), axis=1)

    return preds, df

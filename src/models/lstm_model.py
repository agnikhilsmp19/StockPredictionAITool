import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from src.utils.data_loader import load_and_clean_stock_data

from pathlib import Path

# Set your repo root (relative to this file)
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PRED = ROOT / "data" / "predictions"
DATA_PRED.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
raw_file = DATA_RAW / f"INFY.csv"

# Load your stock data
ticker = "INFY"
data = load_and_clean_stock_data(ticker)

#data = pd.read_csv(raw_file, index_col=0, parse_dates=True)
close_prices = data["Close"].values.reshape(-1, 1)

# Scale data to [0,1]
scaler = MinMaxScaler()
close_scaled = scaler.fit_transform(close_prices)

# Function to create sequences
def create_sequences(series, window=30):
    X = []
    y = []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

window_size = 30
X, y = create_sequences(close_scaled, window=window_size)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=1)

# Predict
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# After y_pred_rescaled and y_test_rescaled are computed
# If your y_test comes from a DataFrame with a date index, use those dates as the index
dates = data.index[-len(y_test_rescaled):]  # Adjust as per your split logic

pred_df = pd.DataFrame({
    "Actual": y_test_rescaled.flatten(),
    "Predicted": y_pred_rescaled.flatten()
}, index=dates)

# Save to the predictions folder, using a root-relative path
ROOT = Path(__file__).resolve().parents[2]
DATA_PRED = ROOT / "data" / "predictions"
DATA_PRED.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

# Example filename pattern: lstm_pred_INFy.csv
pred_path = DATA_PRED / "lstm_pred_INFY.csv"
pred_df.to_csv(pred_path)

#print(f"Saved LSTM predictions to: {pred_path}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_test_rescaled)), y_test_rescaled, label='Actual')
plt.plot(np.arange(len(y_pred_rescaled)), y_pred_rescaled, label='LSTM Prediction')
plt.legend()
plt.title("LSTM Stock Price Prediction (Test Set)")
plt.show()

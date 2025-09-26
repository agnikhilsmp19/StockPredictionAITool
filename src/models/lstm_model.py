import pandas as pd
import numpy as np
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from src.utils.data_loader import load_and_clean_stock_data_lstm

def train_and_save_lstm(ticker, window_size=30, epochs=15, batch_size=16):
    # Set repo paths (always resolves from this script)
    ROOT = Path(__file__).resolve().parents[2]
    DATA_PRED = ROOT / "data" / "predictions"
    DATA_PRED.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    # --- Load and Clean Data ---
    data = load_and_clean_stock_data_lstm(ticker)
    close_prices = data["Close"].values.reshape(-1, 1)
    
    # --- Scale data ---
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close_prices)
    
    # --- Create sequences ---
    def create_sequences(series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i+window])
            y.append(series[i+window])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(close_scaled, window=window_size)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # --- Build LSTM model ---
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # --- Predict & inverse scale ---
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)
    
    # --- Match dates to predictions ---
    dates = data.index[-len(y_test_rescaled):]  # Last test-window indices
    
    pred_df = pd.DataFrame({
        "Actual": y_test_rescaled.flatten(),
        "Predicted": y_pred_rescaled.flatten()
    }, index=dates)
    
    # --- Save CSV dynamically by ticker ---
    pred_path = DATA_PRED / f"lstm_pred_{ticker.upper()}.csv"
    pred_df.to_csv(pred_path)
    print(f"Saved LSTM predictions for {ticker.upper()} to: {pred_path}")

    # --- (Optional) Plot for validation ---
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(y_test_rescaled)), y_test_rescaled, label='Actual')
    plt.plot(np.arange(len(y_pred_rescaled)), y_pred_rescaled, label='LSTM Prediction')
    plt.legend()
    plt.title(f"LSTM Stock Price Prediction ({ticker.upper()} Test Set)")
    plt.show()
    return pred_df

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "INFY"
    train_and_save_lstm(ticker)

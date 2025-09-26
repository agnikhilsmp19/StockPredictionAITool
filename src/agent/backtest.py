import os
import pandas as pd

from src.agent.decision_engine import decide_action

root_dir = os.getcwd()   # Gets current working directory
print("Root directory:", root_dir)
DATA_DIR = os.path.join(root_dir, "data", "raw")

data_path = os.path.join(DATA_DIR, "INFY.csv")

data = pd.read_csv(data_path)
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

cash = 100000  # Initial capital
stock = 0

for i in range(len(data)):
    current = data['Close'].iloc[i]
    predicted = data['Target'].iloc[i]
    action = decide_action(current, predicted)
    
    if action == "Buy" and cash > 0:
        stock += cash / current
        cash = 0
    elif action == "Sell" and stock > 0:
        cash += stock * current
        stock = 0

#portfolio_value = cash + stock * data['Close'].iloc[-1]

# Ensure numeric columns
for col in ["Open", "High", "Low", "Close", "Volume"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Now this will work without error
portfolio_value = cash + stock * data['Close'].iloc[-1]

print("Final Portfolio Value:", portfolio_value)

def backtest(data, initial_cash=100000):
    cash = initial_cash
    stock = 0
    trades = []

    for i in range(len(data) - 1):
        decision = data["Decision"].iloc[i]
        price = data["Close"].iloc[i]

        if decision == "BUY" and cash >= price:
            stock += 1
            cash -= price
            trades.append(f"BUY at {price:.2f}")

        elif decision == "SELL" and stock > 0:
            stock -= 1
            cash += price
            trades.append(f"SELL at {price:.2f}")

    final_value = cash + stock * data["Close"].iloc[-1]
    return final_value, trades[-5:]  # last 5 trades
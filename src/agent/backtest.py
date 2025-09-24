import os
import pandas as pd
from decision_engine import decide_action

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

portfolio_value = cash + stock * data['Close'].iloc[-1]
print("Final Portfolio Value:", portfolio_value)

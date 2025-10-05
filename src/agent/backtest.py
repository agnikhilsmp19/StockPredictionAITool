import os
import pandas as pd

from src.agent.decision_engine import decide_action
from datetime import date
from pathlib import Path
from src.utils.paths import DATA_RAW, DATA_PRED
root_dir = os.getcwd()   # Gets current working directory
#print("Root directory:", root_dir)
DATA_DIR = os.path.join(root_dir, "data", "raw")
end_date = date.today().isoformat()
data_path = os.path.join(DATA_RAW, f"INFY_{end_date}.csv")
print("data_path",data_path)

def backtest(data, price_column="Close"):
    cash = 100000
    stock = 0
    trades = []

    for i in range(len(data) - 1):
        decision = data["Decision"].iloc[i]
        price = data[price_column].iloc[i]

        if decision == "BUY" and cash >= price:
            stock += 1
            cash -= price
            trades.append(f"BUY at {price.round(2)}")
        elif decision == "SELL" and stock > 0:
            stock -= 1
            cash += price
            trades.append(f"SELL at {price.round(2)}")

    final_value = cash + stock * data[price_column].iloc[-1]
    return final_value, trades
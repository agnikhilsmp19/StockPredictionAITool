import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

root_dir = os.getcwd()   # Gets current working directory
print("Root directory:", root_dir)
DATA_DIR = os.path.join(root_dir, "data", "raw")

data_path = os.path.join(DATA_DIR, "INFY.csv")
data = pd.read_csv(data_path, index_col=0, parse_dates=True)
data["Target"] = data["Close"].shift(-1)
data.dropna(inplace=True)

X = data[["Close"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
# rmse = mean_squared_error(y_test, pred, squared=False)
# print("RMSE:", rmse)

rmse = root_mean_squared_error(y_test, pred)
print("RMSE:", rmse)

plt.figure(figsize=(10,6))
plt.plot(y_test.values, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.show()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

def run_linear_regression(data):
    data = data.copy()
    data["Target"] = data["Close"].shift(-1)
    data.dropna(inplace=True)
    X = data[["Close"]]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}, index=y_test.index)
    data_test = X_test.copy()
    data_test["Predicted"] = y_pred
    return rmse, pred_df, data_test

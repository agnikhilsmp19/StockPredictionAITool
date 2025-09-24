from fastapi import FastAPI

app = FastAPI()

@app.get("/predict")
def predict(stock: str = "AAPL"):
    return {"stock": stock, "prediction": 150, "action": "Hold"}

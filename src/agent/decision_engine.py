from src.agent.indicators import calculate_rsi

def make_decision(preds, df, threshold=0.02):
    """
    preds: list of predicted prices (from LSTM)
    df: stock DataFrame with Close prices
    threshold: % change to trigger decision
    """

    explanation = []

    # --- 1. Prediction Trend ---
    if len(preds) < 2:
        return "Not enough data", "Prediction data insufficient"
    
    change = (preds[-1] - preds[0]) / preds[0]
    if change > threshold:
        pred_signal = "BUY"
        explanation.append(f"Model predicts upward trend ({change*100:.2f}%).")
    elif change < -threshold:
        pred_signal = "SELL"
        explanation.append(f"Model predicts downward trend ({change*100:.2f}%).")
    else:
        pred_signal = "HOLD"
        explanation.append("Model predicts stable movement.")
    
    # --- 2. RSI ---
    rsi = calculate_rsi(df["Close"])
    latest_rsi = rsi.iloc[-1].values[0]
    #latest_rsi = df["RSI"].iloc[-1]   # gets the last value as a float
    print("latest_rsi - ",latest_rsi)
    if latest_rsi > 70:
        rsi_signal = "SELL"
        explanation.append(f"RSI = {latest_rsi:.2f} → Overbought.")
    elif latest_rsi < 30:
        rsi_signal = "BUY"
        explanation.append(f"RSI = {latest_rsi:.2f} → Oversold.")
    else:
        rsi_signal = "HOLD"
        explanation.append(f"RSI = {latest_rsi:.2f} → Neutral zone.")
    
    # --- 3. SMA Crossover ---
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    if df["SMA50"].iloc[-1] > df["SMA200"].iloc[-1]:
        sma_signal = "BUY"
        explanation.append("SMA50 crossed above SMA200 → Bullish signal.")
    else:
        sma_signal = "SELL"
        explanation.append("SMA50 below SMA200 → Bearish signal.")

    # --- Final Decision (Majority Voting) ---
    signals = [pred_signal, rsi_signal, sma_signal]
    final = max(set(signals), key=signals.count)

    return final, explanation

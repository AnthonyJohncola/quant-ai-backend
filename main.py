from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

app = FastAPI()

# ---------------------------
# DATA FUNCTIONS
# ---------------------------

def get_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    df["rsi"] = RSIIndicator(df["Close"]).rsi()
    df["macd"] = MACD(df["Close"]).macd()
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    df.dropna(inplace=True)
    return df

# ---------------------------
# ANALYTICS FUNCTIONS
# ---------------------------

def monte_carlo_simulation(last_price, mu, sigma, days=30, simulations=300):
    results = []

    for _ in range(simulations):
        prices = [last_price]
        for _ in range(days):
            shock = np.random.normal(mu, sigma)
            prices.append(prices[-1] * (1 + shock))
        results.append(prices)

    return results

def sharpe_ratio(returns):
    if np.std(returns) == 0:
        return 0
    return float(np.mean(returns) / np.std(returns))

def max_drawdown(prices):
    peak = prices[0]
    max_dd = 0

    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        if drawdown > max_dd:
            max_dd = drawdown

    return float(max_dd)

# ---------------------------
# API ENDPOINT
# ---------------------------

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    try:
        df = get_data(ticker.upper())
        df = add_indicators(df)

        last_price = float(df["Close"].iloc[-1])
        mu = df["returns"].mean()
        sigma = df["returns"].std()

        simulations = monte_carlo_simulation(last_price, mu, sigma)

        sharpe = sharpe_ratio(df["returns"])
        drawdown = max_drawdown(df["Close"].values)

        confidence_95 = np.percentile(
            [sim[-1] for sim in simulations], [2.5, 97.5]
        )

        return {
            "ticker": ticker.upper(),
            "last_price": last_price,
            "sharpe_ratio": sharpe,
            "max_drawdown": drawdown,
            "expected_return_daily": float(mu),
            "volatility_daily": float(sigma),
            "confidence_interval_95": {
                "lower": float(confidence_95[0]),
                "upper": float(confidence_95[1])
            },
            "monte_carlo_sample_paths": simulations[:25]
        }

    except Exception as e:
        return {"error": str(e)}

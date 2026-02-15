from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For now allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# DATA FUNCTIONS
# ---------------------------

def get_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")

    # Fix for yfinance multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
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
        import numpy as np

    # ==============================
    # Log Returns
    # ==============================
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    mean_return = returns.mean()
    volatility = returns.std()

    # ==============================
    # Annualized Metrics
    # ==============================
    annual_return = mean_return * 252
    annual_volatility = volatility * np.sqrt(252)

    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility != 0
        else 0
    )

    # ==============================
    # 30-Day Monte Carlo Simulation
    # ==============================
    days = 30
    simulations = []

    last_price = df["Close"].iloc[-1]

    for _ in range(1000):
        daily_returns = np.random.normal(mean_return, volatility, days)
        price_path = last_price * np.exp(np.cumsum(daily_returns))
        simulations.append(price_path)

    simulations = np.array(simulations)
    final_prices = simulations[:, -1]

    expected_price = np.mean(final_prices)
    probability_up = np.mean(final_prices > last_price)
    var_95 = np.percentile(final_prices, 5)

    # ==============================
    # Directional Bias
    # ==============================
    if probability_up > 0.55:
        bias = "Bullish"
    elif probability_up < 0.45:
        bias = "Bearish"
    else:
        bias = "Neutral"

    # ==============================
    # Return API Response
    # ==============================
    return {
        "last_price": float(last_price),
        "expected_30d_price": float(expected_price),
        "probability_upside": float(probability_up),
        "value_at_risk_95": float(var_95),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "bias": bias,
        "confidence_interval": [
            float(np.percentile(final_prices, 2.5)),
            float(np.percentile(final_prices, 97.5))
        ],
        "simulations": simulations.tolist()
    } import numpy as np

    # ==============================
    # Log Returns
    # ==============================
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    mean_return = returns.mean()
    volatility = returns.std()

    # ==============================
    # Annualized Metrics
    # ==============================
    annual_return = mean_return * 252
    annual_volatility = volatility * np.sqrt(252)

    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility != 0
        else 0
    )

    # ==============================
    # 30-Day Monte Carlo Simulation
    # ==============================
    days = 30
    simulations = []

    last_price = df["Close"].iloc[-1]

    for _ in range(1000):
        daily_returns = np.random.normal(mean_return, volatility, days)
        price_path = last_price * np.exp(np.cumsum(daily_returns))
        simulations.append(price_path)

    simulations = np.array(simulations)
    final_prices = simulations[:, -1]

    expected_price = np.mean(final_prices)
    probability_up = np.mean(final_prices > last_price)
    var_95 = np.percentile(final_prices, 5)

    # ==============================
    # Directional Bias
    # ==============================
    if probability_up > 0.55:
        bias = "Bullish"
    elif probability_up < 0.45:
        bias = "Bearish"
    else:
        bias = "Neutral"

    # ==============================
    # Return API Response
    # ==============================
    return {
        "last_price": float(last_price),
        "expected_30d_price": float(expected_price),
        "probability_upside": float(probability_up),
        "value_at_risk_95": float(var_95),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "bias": bias,
        "confidence_interval": [
            float(np.percentile(final_prices, 2.5)),
            float(np.percentile(final_prices, 97.5))
        ],
        "simulations": simulations.tolist()
    }

    except Exception as e:
        return {"error": str(e)}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

app = FastAPI()

# ---------------------------
# CORS CONFIG
# ---------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    df.dropna(inplace=True)
    return df


# ---------------------------
# API ENDPOINT
# ---------------------------

@app.get("/analyze/{ticker}")
def analyze(ticker: str):
    try:
        ticker = ticker.upper()

        df = get_data(ticker)
        df = add_indicators(df)

        # ==============================
        # LOG RETURNS MODEL
        # ==============================
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        mean_return = returns.mean()
        volatility = returns.std()

        annual_return = mean_return * 252
        annual_volatility = volatility * np.sqrt(252)

        sharpe_ratio = (
            annual_return / annual_volatility
            if annual_volatility != 0
            else 0
        )

        # ==============================
        # MONTE CARLO (30 DAYS)
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
        probability_up = float(np.mean(final_prices > last_price))
        var_95 = float(np.percentile(final_prices, 5))

        # ==============================
        # DIRECTIONAL BIAS
        # ==============================
        if probability_up > 0.55:
            bias = "Bullish"
        elif probability_up < 0.45:
            bias = "Bearish"
        else:
            bias = "Neutral"

        # ==============================
        # RESPONSE
        # ==============================
        return {
            "ticker": ticker,
            "last_price": float(last_price),
            "expected_30d_price": float(expected_price),
            "probability_upside": probability_up,
            "value_at_risk_95": var_95,
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "rsi": float(df["rsi"].iloc[-1]),
            "macd": float(df["macd"].iloc[-1]),
            "bias": bias,
            "confidence_interval_95": {
                "lower": float(np.percentile(final_prices, 2.5)),
                "upper": float(np.percentile(final_prices, 97.5))
            },
            "simulations": simulations.tolist()
        }

    except Exception as e:
        return {"error": str(e)}


    except Exception as e:
        return {"error": str(e)}

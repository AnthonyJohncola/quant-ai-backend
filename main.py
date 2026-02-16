from fastapi import FastAPI, Query
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
def analyze(ticker: str, lite: bool = Query(False)):

    try:
        ticker = ticker.upper()

        df = get_data(ticker)
        df = add_indicators(df)

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
        # REGIME DETECTION
        # ==============================

        df["sma_50"] = df["Close"].rolling(50).mean()
        df["sma_200"] = df["Close"].rolling(200).mean()

        if df["sma_50"].iloc[-1] > df["sma_200"].iloc[-1]:
            trend_regime = "Uptrend"
            trend_score = 1
        elif df["sma_50"].iloc[-1] < df["sma_200"].iloc[-1]:
            trend_regime = "Downtrend"
            trend_score = 0
        else:
            trend_regime = "Range"
            trend_score = 0.5

        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        median_vol = rolling_vol.median()

        volatility_regime = (
            "High Volatility" if current_vol > median_vol else "Low Volatility"
        )

        # ==============================
        # MONTE CARLO
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

        if probability_up > 0.55:
            bias = "Bullish"
        elif probability_up < 0.45:
            bias = "Bearish"
        else:
            bias = "Neutral"

        # ==============================
        # SIGNAL SCORING
        # ==============================

        prob_score = probability_up
        sharpe_score = min(max((sharpe_ratio + 1) / 2, 0), 1)

        rsi = df["rsi"].iloc[-1]
        if rsi < 30:
            rsi_score = 1
        elif rsi > 70:
            rsi_score = 0
        else:
            rsi_score = 1 - abs(rsi - 50) / 20
            rsi_score = min(max(rsi_score, 0), 1)

        macd = df["macd"].iloc[-1]
        macd_score = 1 if macd > 0 else 0

        signal_score = (
            prob_score * 0.30 +
            sharpe_score * 0.20 +
            rsi_score * 0.15 +
            macd_score * 0.15 +
            trend_score * 0.20
        ) * 100

        if signal_score > 75:
            signal_label = "Strong Buy"
        elif signal_score > 60:
            signal_label = "Buy"
        elif signal_score > 40:
            signal_label = "Neutral"
        elif signal_score > 25:
            signal_label = "Sell"
        else:
            signal_label = "Strong Sell"

        response = {
            "ticker": ticker,
            "last_price": float(last_price),
            "expected_30d_price": float(expected_price),
            "probability_upside": probability_up,
            "value_at_risk_95": var_95,
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "rsi": float(rsi),
            "macd": float(macd),
            "trend_regime": trend_regime,
            "volatility_regime": volatility_regime,
            "bias": bias,
            "signal_score": float(signal_score),
            "signal_label": signal_label,
            "confidence_interval_95": {
                "lower": float(np.percentile(final_prices, 2.5)),
                "upper": float(np.percentile(final_prices, 97.5))
            }
        }

        # Only include simulations if NOT lite
        if not lite:
            response["simulations"] = simulations.tolist()

        return response

    except Exception as e:
        return {"error": str(e)}

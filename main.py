import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# Explicit CORS (works reliably with Railway)
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================
# CONFIG
# ============================

MONTE_CARLO_SIMS = 500
MONTE_CARLO_DAYS = 60


# ============================
# UTIL FUNCTIONS
# ============================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi.iloc[-1])


def monte_carlo_simulation(last_price, daily_volatility):
    simulations = []

    for _ in range(MONTE_CARLO_SIMS):
        price_path = [float(last_price)]

        for _ in range(MONTE_CARLO_DAYS):
            shock = np.random.normal(0, daily_volatility)
            price = price_path[-1] * (1 + shock)
            price_path.append(float(price))

        simulations.append(price_path)

    return simulations


def score_signal(rsi, macd):
    score = 50

    if rsi < 30:
        score += 20
    elif rsi > 70:
        score -= 20

    if macd > 0:
        score += 10
    else:
        score -= 10

    score = max(0, min(100, score))

    if score > 65:
        label = "Bullish"
    elif score < 35:
        label = "Bearish"
    else:
        label = "Neutral"

    return float(score), label


# ============================
# ROUTES
# ============================

@app.route("/")
def home():
    return "Quant AI Backend Running"


@app.route("/analyze/<ticker>")
def analyze(ticker):

    lite_mode = request.args.get("lite", "false").lower() == "true"

    try:
        data = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=True
        )

        if data is None or data.empty:
            return jsonify({"error": "No data found"}), 404

        # Handle MultiIndex safely
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data["Close"][ticker.upper()]
        else:
            close_prices = data["Close"]

        close_prices = close_prices.dropna()

        if close_prices.empty:
            return jsonify({"error": "No valid price data"}), 404

        # RSI
        rsi = calculate_rsi(close_prices)

        # MACD (FORCED FLOAT)
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()

        macd_series = ema_12 - ema_26
        macd = float(macd_series.iloc[-1])

        daily_volatility = float(close_prices.pct_change().std())

        score, label = score_signal(rsi, macd)

        trend_regime = "Bullish Trend" if macd > 0 else "Bearish Trend"

        response = {
            "ticker": ticker.upper(),
            "signal_score": score,
            "signal_label": label,
            "trend_regime": trend_regime,
            "rsi": rsi,
            "macd": macd
        }

        # Monte Carlo only if NOT lite
        if not lite_mode:
            simulations = monte_carlo_simulation(
                close_prices.iloc[-1],
                daily_volatility
            )
            response["simulations"] = simulations

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================
# RUN
# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

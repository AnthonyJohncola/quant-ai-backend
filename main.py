import numpy as np
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
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

    return rsi.iloc[-1]


def monte_carlo_simulation(last_price, daily_volatility):
    simulations = []

    for _ in range(MONTE_CARLO_SIMS):
        price_path = [last_price]

        for _ in range(MONTE_CARLO_DAYS):
            shock = np.random.normal(0, daily_volatility)
            price = price_path[-1] * (1 + shock)
            price_path.append(price)

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

    return score, label


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
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)

        if data.empty:
            return jsonify({"error": "No data found"}), 404

        close_prices = data["Close"]

        # Indicators
        rsi = calculate_rsi(close_prices)

        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        macd = (ema_12 - ema_26).iloc[-1]

        daily_volatility = close_prices.pct_change().std()

        score, label = score_signal(rsi, macd)

        trend_regime = "Bullish Trend" if macd > 0 else "Bearish Trend"

        response = {
            "ticker": ticker.upper(),
            "signal_score": float(score),
            "signal_label": label,
            "trend_regime": trend_regime,
            "rsi": float(rsi),
            "macd": float(macd),
        }

        # Only include simulations if NOT lite
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
# START SERVER
# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import AverageTrueRange


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Market Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 AI Market Predictor")
st.caption(
    "Educational market-analysis and paper-trading tool. "
    "Predictions are probabilities, not guarantees."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Market Settings")

market_type = st.sidebar.selectbox(
    "Market",
    [
        "Stock",
        "Crypto",
        "Forex",
        "Upload CSV"
    ]
)

symbol = st.sidebar.text_input(
    "Symbol",
    "AAPL"
)

period = st.sidebar.selectbox(
    "Historical Data",
    [
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y"
    ]
)

interval = st.sidebar.selectbox(
    "Timeframe",
    [
        "1d",
        "1h"
    ]
)

lookahead = st.sidebar.slider(
    "Prediction horizon (candles)",
    1,
    20,
    1
)

threshold = st.sidebar.slider(
    "Minimum probability",
    0.50,
    0.90,
    0.60,
    0.01
)


# =========================================================
# DATA
# =========================================================

@st.cache_data
def download_data(symbol, period, interval):

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        return None

    # Handle multi-index columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }
    )

    data = data[
        ["open", "high", "low", "close", "volume"]
    ]

    return data.dropna()


# =========================================================
# CSV LOADER
# =========================================================

uploaded_file = None

if market_type == "Upload CSV":

    uploaded_file = st.sidebar.file_uploader(
        "Upload OHLCV CSV",
        type=["csv"]
    )

    if uploaded_file:

        data = pd.read_csv(uploaded_file)

        data.columns = [
            x.lower().strip()
            for x in data.columns
        ]

        required = [
            "open",
            "high",
            "low",
            "close"
        ]

        missing = [
            x for x in required
            if x not in data.columns
        ]

        if missing:

            st.error(
                f"Missing columns: {missing}"
            )

            st.stop()

        if "volume" not in data.columns:
            data["volume"] = 0

        data = data[
            ["open", "high", "low", "close", "volume"]
        ].dropna()

    else:

        st.info(
            "Upload an OHLCV CSV file to continue."
        )

        st.stop()

else:

    # Some examples
    if market_type == "Crypto":
        st.sidebar.info(
            "Example: BTC-USD, ETH-USD"
        )

    elif market_type == "Forex":
        st.sidebar.info(
            "Example: EURUSD=X, GBPUSD=X"
        )

    elif market_type == "Stock":
        st.sidebar.info(
            "Example: AAPL, TSLA, NVDA"
        )

    data = download_data(
        symbol,
        period,
        interval
    )

    if data is None:

        st.error(
            "No market data found. Check the symbol."
        )

        st.stop()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(df):

    df = df.copy()

    # Returns
    df["return"] = df["close"].pct_change()

    # Moving averages
    df["sma20"] = SMAIndicator(
        df["close"],
        window=20
    ).sma_indicator()

    df["sma50"] = SMAIndicator(
        df["close"],
        window=50
    ).sma_indicator()

    df["ema20"] = EMAIndicator(
        df["close"],
        window=20
    ).ema_indicator()

    df["ema50"] = EMAIndicator(
        df["close"],
        window=50
    ).ema_indicator()

    # RSI
    df["rsi"] = RSIIndicator(
        df["close"],
        window=14
    ).rsi()

    # MACD
    macd = MACD(
        df["close"]
    )

    df["macd"] = macd.macd()

    df["macd_signal"] = (
        macd.macd_signal()
    )

    df["macd_hist"] = (
        macd.macd_diff()
    )

    # ATR
    atr = AverageTrueRange(
        df["high"],
        df["low"],
        df["close"],
        window=14
    )

    df["atr"] = atr.average_true_range()

    # Price position
    df["price_vs_sma20"] = (
        df["close"] / df["sma20"] - 1
    )

    df["price_vs_sma50"] = (
        df["close"] / df["sma50"] - 1
    )

    # Momentum
    df["momentum_5"] = (
        df["close"].pct_change(5)
    )

    df["momentum_10"] = (
        df["close"].pct_change(10)
    )

    # Volatility
    df["volatility"] = (
        df["return"]
        .rolling(20)
        .std()
    )

    return df


data = create_features(data)


# =========================================================
# TARGET
# =========================================================

data["future_close"] = (
    data["close"].shift(-lookahead)
)

data["future_return"] = (
    data["future_close"] /
    data["close"] - 1
)


# Classification:
#
# 1 = UP
# 0 = DOWN
#
# A more advanced version can add NEUTRAL
# based on volatility.


data["target"] = (
    data["future_return"] > 0
).astype(int)


# =========================================================
# FEATURES
# =========================================================

features = [
    "return",
    "sma20",
    "sma50",
    "ema20",
    "ema50",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr",
    "price_vs_sma20",
    "price_vs_sma50",
    "momentum_5",
    "momentum_10",
    "volatility"
]

model_data = data.dropna().copy()


if len(model_data) < 200:

    st.error(
        "Not enough historical data for reliable training."
    )

    st.stop()


X = model_data[features]
y = model_data["target"]


# =========================================================
# TIME-SERIES SPLIT
# =========================================================

split = int(
    len(model_data) * 0.80
)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced_subsample"
)

model.fit(
    X_train_scaled,
    y_train
)


# =========================================================
# TEST
# =========================================================

predictions = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    predictions
)


# =========================================================
# CURRENT PREDICTION
# =========================================================

latest = model_data.iloc[-1]

latest_X = pd.DataFrame(
    [latest[features]],
    columns=features
)

latest_scaled = scaler.transform(
    latest_X
)

probabilities = model.predict_proba(
    latest_scaled
)[0]

down_probability = probabilities[0]
up_probability = probabilities[1]


if up_probability >= threshold:

    prediction = "🟢 UP"

elif down_probability >= threshold:

    prediction = "🔴 DOWN"

else:

    prediction = "⚪ NEUTRAL"


# =========================================================
# DASHBOARD
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Price",
        f"{latest['close']:.4f}"
    )

with col2:
    st.metric(
        "AI Prediction",
        prediction
    )

with col3:
    st.metric(
        "UP Probability",
        f"{up_probability * 100:.2f}%"
    )

with col4:
    st.metric(
        "Model Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )


# =========================================================
# PROBABILITY BAR
# =========================================================

st.subheader("AI Prediction Probability")

prob_df = pd.DataFrame(
    {
        "Direction": [
            "DOWN",
            "UP"
        ],
        "Probability": [
            down_probability * 100,
            up_probability * 100
        ]
    }
)

st.bar_chart(
    prob_df.set_index("Direction")
)


# =========================================================
# CANDLESTICK CHART
# =========================================================

st.subheader("Market Chart")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name="Price"
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["sma20"],
        name="SMA 20"
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["sma50"],
        name="SMA 50"
    )
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# INDICATORS
# =========================================================

st.subheader("Technical Indicators")

i1, i2, i3, i4 = st.columns(4)

with i1:

    st.metric(
        "RSI",
        f"{latest['rsi']:.2f}"
    )

with i2:

    st.metric(
        "MACD",
        f"{latest['macd']:.4f}"
    )

with i3:

    st.metric(
        "EMA 20",
        f"{latest['ema20']:.4f}"
    )

with i4:

    st.metric(
        "ATR",
        f"{latest['atr']:.4f}"
    )


# =========================================================
# MODEL REPORT
# =========================================================

st.subheader("Model Evaluation")

report = classification_report(
    y_test,
    predictions,
    target_names=[
        "DOWN",
        "UP"
    ],
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader(
    "What the AI Used"
)

importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": model.feature_importances_
    }
).sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)


# =========================================================
# LAST DATA
# =========================================================

st.subheader(
    "Latest Market Data"
)

st.dataframe(
    data.tail(20),
    use_container_width=True
)


# =========================================================
# DISCLAIMER
# =========================================================

st.warning(
    "This system is an experimental prediction model. "
    "It does not know the future and can be wrong. "
    "Historical accuracy does not guarantee future performance. "
    "Use paper trading/backtesting for evaluation."
)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from streamlit_autorefresh import st_autorefresh


st.set_page_config(
    page_title="AI Market & Sentiment Predictor",
    page_icon="📈",
    layout="centered"
) 

st_autorefresh(
    interval=300000,
    key="market_refresh"
)

st.title("🤖 AI Market & Sentiment Predictor")
st.caption(
    "Educational tool: technical data and news sentiment are probabilities, "
    "not guaranteed investment advice."
)

market = st.sidebar.selectbox(
    "Market",
    [
        "Indian NSE",
        "Indian BSE",
        "US Market"
    ]
)

if market == "Indian NSE":

    user_symbol = st.sidebar.text_input(
        "NSE Symbol",
        "RELIANCE"
    ).upper().strip()

    symbol = user_symbol + ".NS"

elif market == "Indian BSE":

    user_symbol = st.sidebar.text_input(
        "BSE Symbol",
        "RELIANCE"
    ).upper().strip()

    symbol = user_symbol + ".BO"

else:

    symbol = st.sidebar.text_input(
        "US Symbol",
        "AAPL"
    ).upper().strip()

st.sidebar.info(
    f"Using symbol: {symbol}"
)


period = st.sidebar.selectbox(
    "Historical Period",
    ["6mo", "1y", "2y", "5y"]
)

interval = st.sidebar.selectbox(
    "Prediction Timeframe",
    [
        "5m",
        "15m",
        "1d"
    ],
    index=1
)

threshold = st.sidebar.slider(
    "Minimum probability",
    0.50,
    0.90,
    0.60,
    0.01
)

@st.cache_data(ttl=300)
data = download_data(
    symbol,
    interval,
    period
)
    if interval in ["5m", "15m"]:
        request_period = "60d"
    else:
        request_period = selected_period

    data = yf.download(
        symbol,
        period=request_period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        return None

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

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        return None

    return data[required].dropna()

data = download_data(
    symbol,
    interval,
    period
)

@st.cache_data(ttl=900)
def get_news(symbol):

    try:
        news_items = yf.Ticker(symbol).news
    except Exception:
        return []

    results = []

    for item in news_items[:20]:

        content = item.get("content", item)

        title = (
            content.get("title")
            or item.get("title")
            or ""
        )

        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else item.get("publisher", "")
        )

        link = ""

        if isinstance(content.get("clickThroughUrl"), dict):
            link = content["clickThroughUrl"].get("url", "")

        if not link:
            link = item.get("link", "")

        if title:
            results.append(
                {
                    "title": title,
                    "publisher": publisher or "Unknown",
                    "link": link
                }
            )

    return results


def create_features(data):

    df = data.copy()

    df["return"] = df["close"].pct_change()

    df["sma20"] = SMAIndicator(
        df["close"], window=20
    ).sma_indicator()

    df["sma50"] = SMAIndicator(
        df["close"], window=50
    ).sma_indicator()

    df["ema20"] = EMAIndicator(
        df["close"], window=20
    ).ema_indicator()

    df["rsi"] = RSIIndicator(
        df["close"], window=14
    ).rsi()

    macd = MACD(df["close"])

    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    atr = AverageTrueRange(
        df["high"],
        df["low"],
        df["close"],
        window=14
    )

    df["atr"] = atr.average_true_range()

    df["price_vs_sma20"] = (
        df["close"] / df["sma20"] - 1
    )

    df["momentum_5"] = df["close"].pct_change(5)

    df["volatility"] = (
        df["return"].rolling(20).std()
    )

    return df


def get_sentiment(news):

    analyzer = SentimentIntensityAnalyzer()

    scores = []
    rows = []

    for item in news:

        score = analyzer.polarity_scores(
            item["title"]
        )["compound"]

        scores.append(score)

        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        rows.append(
            {
                "Headline": item["title"],
                "Publisher": item["publisher"],
                "Sentiment": label,
                "Score": round(score, 3),
                "Link": item["link"]
            }
        )

    if not scores:
        return 0.0, pd.DataFrame()

    average_score = float(np.mean(scores))

    return average_score, pd.DataFrame(rows)

if data is None:
    st.error(
        "Data nahi mila. Symbol check karo, jaise AAPL, TSLA, "
        "BTC-USD ya EURUSD=X."
    )
    st.stop()


data = create_features(data)

features = [
    "return",
    "sma20",
    "sma50",
    "ema20",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr",
    "price_vs_sma20",
    "momentum_5",
    "volatility"
]

data["future_return"] = (
    data["close"].shift(-1) /
    data["close"] - 1
)

up_limit = 0.0015
down_limit = -0.0015

data["target"] = np.select(
    [
        data["future_return"] > up_limit,
        data["future_return"] < down_limit
    ],
    [
        1,
        0
    ],
    default=2
)

model_data = data.dropna().copy()

if len(model_data) < 150:
    st.error(
        "There is insufficient historical data for training. "
        "Change the timeframe or period."
    )
    st.stop()

X = model_data[features]
y = model_data["target"]

split = int(len(model_data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

latest = model_data.iloc[-1]

latest_x = scaler.transform(
    pd.DataFrame(
        [latest[features]],
        columns=features
    )
)

probabilities = model.predict_proba(
    latest_x
)[0]

class_probabilities = dict(
    zip(model.classes_, probabilities)
)

down_probability = class_probabilities.get(0, 0)
up_probability = class_probabilities.get(1, 0)
neutral_probability = class_probabilities.get(2, 0)

if up_probability >= threshold:
    prediction = "🟢 UP"
elif down_probability >= threshold:
    prediction = "🔴 DOWN"
else:
    prediction = "⚪ NEUTRAL"

news = get_news(symbol)
news_score, news_df = get_sentiment(news)

news_up_probability = 0.50 + (
    news_score * 0.35
)

news_up_probability = float(
    np.clip(news_up_probability, 0.05, 0.95)
)

combined_up = (
    up_probability * 0.70
    + news_up_probability * 0.30
)

combined_down = 1 - combined_up

if combined_up >= threshold:
    final_signal = "🟢 UP"
elif combined_down >= threshold:
    final_signal = "🔴 DOWN"
else:
    final_signal = "⚪ NEUTRAL"


st.subheader("Prediction")

st.info(
    f"Market: {market} | Symbol: {symbol} | "
    f"Timeframe: {interval} | Auto-refresh: 5 minutes"
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Current Price",
        f"{latest['close']:.2f}"
    )

with col2:
    st.metric(
        "Signal",
        final_signal
    )

with col3:
    st.metric(
        "UP Probability",
        f"{up_probability * 100:.1f}%"
    )

with col4:
    st.metric(
        "DOWN Probability",
        f"{down_probability * 100:.1f}%"
    )

with col5:
    st.metric(
        "NEUTRAL Probability",
        f"{neutral_probability * 100:.1f}%"
    )

st.subheader("Signal Breakdown")

breakdown = pd.DataFrame(
    {
        "Signal": [
            "Technical UP",
            "News-based UP",
            "Combined UP"
        ],
        "Probability": [
            up_probability * 100,
            news_up_probability * 100,
            combined_up * 100
        ]
    }
)

st.bar_chart(
    breakdown.set_index("Signal")
)


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
    height=500,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.subheader("Latest News Sentiment")

if news_df.empty:
    st.info(
"No news was found for this symbol."    )
else:

    st.dataframe(
        news_df.drop(columns=["Link"]),
        use_container_width=True
    )

    for _, row in news_df.iterrows():

        if row["Link"]:
            st.markdown(
                f"- [{row['Headline']}]({row['Link']})"
            )


st.subheader("Technical Indicators")

i1, i2, i3, i4 = st.columns(4)

with i1:
    st.metric("RSI", f"{latest['rsi']:.2f}")

with i2:
    st.metric("MACD", f"{latest['macd']:.4f}")

with i3:
    st.metric("EMA 20", f"{latest['ema20']:.2f}")

with i4:
    st.metric("ATR", f"{latest['atr']:.2f}")


st.warning(
    "This is experimental educational model. "
    "News sentiment signals the current headlines, "
"not a guarantee based on history. Before trading with real money, "
"engage in paper trading and proper backtesting."
 " Thank You. "
)

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
from sklearn.metrics import accuracy_score

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


st.set_page_config(
    page_title="AI Market & Sentiment Predictor",
    page_icon="📈",
    layout="centered"
)

st.title("🤖 AI Market & Sentiment Predictor")
st.caption(
    "Educational tool: technical data and news sentiment are probabilities, "
    "not guaranteed investment advice."
)


symbol = st.sidebar.text_input("Stock/Crypto Symbol", "AAPL").upper().strip()

period = st.sidebar.selectbox(
    "Historical Period",
    ["6mo", "1y", "2y", "5y"]
)

threshold = st.sidebar.slider(
    "Minimum probability",
    0.50,
    0.90,
    0.60,
    0.01
)


@st.cache_data(ttl=900)
def download_data(symbol, period):

    data = yf.download(
        symbol,
        period=period,
        interval="1d",
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

    required = ["open", "high", "low", "close", "volume"]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        return None

    return data[required].dropna()


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


data = download_data(symbol, period)

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
    data["close"].shift(-1) / data["close"] - 1
)

data["target"] = (
    data["future_return"] > 0
).astype(int)

model_data = data.dropna().copy()

if len(model_data) < 150:
    st.error(
        "Training ke liye historical data kam hai. "
        "Period ko 1y ya 2y karo."
    )
    st.stop()


split = int(len(model_data) * 0.80)

X = model_data[features]
y = model_data["target"]

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

technical_probabilities = model.predict_proba(
    latest_x
)[0]

technical_down = technical_probabilities[0]
technical_up = technical_probabilities[1]


news = get_news(symbol)
news_score, news_df = get_sentiment(news)

if news_score > 0.05:
    news_label = "🟢 Positive"
elif news_score < -0.05:
    news_label = "🔴 Negative"
else:
    news_label = "⚪ Neutral"


news_up_probability = 0.50 + (news_score * 0.35)
news_up_probability = float(
    np.clip(news_up_probability, 0.05, 0.95)
)

combined_up = (
    technical_up * 0.70
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

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "Current Price",
        f"{latest['close']:.2f}"
    )

with c2:
    st.metric(
        "Final Signal",
        final_signal
    )

c3, c4, c5 = st.columns(3)

with c3:
    st.metric(
        "Technical UP",
        f"{technical_up * 100:.1f}%"
    )

with c4:
    st.metric(
        "News Sentiment",
        news_label
    )

with c5:
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.1f}%"
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
            technical_up * 100,
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
        "Is symbol ke liye news available nahi mili."
    )
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
    "Ye experimental educational model hai. "
    "News sentiment current headlines ka signal hai, "
    "historical guarantee nahi. Real money se trade karne se pehle "
    "paper trading aur proper backtesting karo."
)

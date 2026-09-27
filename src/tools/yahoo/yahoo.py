import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def sentiment(ticker):
    news = ticker.news
    if not news:
        raise ValueError("Yahoo returned no news")

    frame = pd.DataFrame(item.get("content") for item in news)
    if "title" not in frame.columns:
        raise ValueError("Yahoo news has no title field")
    scores = frame["title"].apply(
        lambda title: analyzer.polarity_scores(title)["compound"]
    )
    return {
        "mean": float(scores.mean()),
        "news": news,
        "price_targets": ticker.analyst_price_targets,
    }


def trading_data(ticker):
    day = ticker.history("1d")
    month = ticker.history("1mo")[["Open", "Volume"]].reset_index()
    year = ticker.history("1y")[["Open", "Volume"]].reset_index()
    latest = ticker.history("5d").iloc[-1]

    return {
        "price": {
            "Day": day,
            "Month": month,
            "Year": year,
            "High": latest["High"],
            "Low": latest["Low"],
            "Open": latest["Open"],
            "Close": latest["Close"],
        },
        "volume": {"1d": day["Volume"], "1mo": month["Volume"], "1y": year["Volume"]},
    }


def retrieve_yahoo_data(ticker):
    yahoo_ticker = yf.Ticker(ticker)
    sentiment_data = sentiment(yahoo_ticker)
    market_data = trading_data(yahoo_ticker)
    price = market_data["price"]

    return {
        "sentiment": sentiment_data,
        "price": {
            "day": price["Day"],
            "month": price["Month"],
            "year": price["Year"],
            "high": price["High"],
            "low": price["Low"],
            "open": price["Open"],
            "close": price["Close"],
        },
        "volume": market_data["volume"],
    }

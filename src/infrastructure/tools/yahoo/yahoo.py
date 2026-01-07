import yfinance as yf
import pandas as pd
from src.util.log_config import setup_logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
logger = setup_logging('yahoo')

def sentiment(ticker: yf.Ticker):
    news = ticker.news
    assert news 
    
    #Retrieve only the content  
    filtered_news = list(map(lambda x: x.get('content'), news))
    
    df = pd.DataFrame(filtered_news)
    assert 'title' in df.columns
    scores = df['title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    return {
        'mean': float(scores.mean()),
        'news': news,
        'price_targets': ticker.analyst_price_targets
        }

def trading_data(ticker: yf.Ticker):
    """Method to extract only the price data needed 
    for the Valuation agent 
    """
    day = ticker.history('1d')
    #Filtering for Date, Open Price and Volume 
    month = ticker.history('1mo')
    mdf = pd.DataFrame(month)
    mdf = mdf[['Open', 'Volume']].reset_index()
    year = ticker.history('1y')

    ydf = pd.DataFrame(year)
    ydf = ydf[['Open', 'Volume']].reset_index()
    
    # Daily price calculation, 5d ensures that latest full day is included 
    range = ticker.history('5d').iloc[-1]

    return {
        'price': {
            'Day': day,      #Price data for a day
            'Month': mdf,   # month
            'Year': ydf,     # year
            'High': range['High'],              # days high
            'Low' : range['Low'],               # days low 
            'Open': range['Open'],              # days open
            'Close': range['Close']             # days close 
        },
        'volume': {
            '1d' : day['Volume'],
            '1mo': mdf['Volume'],
            '1y': ydf['Volume']
        }
    }
    
def retrieve_yahoo_data(ticker: str): 
    yfTicker = yf.Ticker(ticker)
    sentiment_data = sentiment(yfTicker)
    td = trading_data(yfTicker)
    price = td['price']
    volume = td['volume']
    data =  {
        'sentiment': {
            'mean': sentiment_data['mean'],
            'news': sentiment_data['news'],
            'price_targets': sentiment_data['price_targets']
        },
        'price': {
            'day': price['Day'],      #Price data for a day
            'month': price['Month'],   # month
            'year': price['Year'],     # year
            'high': price['High'],              # days high
            'low' : price['Low'],               # days low 
            'open': price['Open'],              # days open
            'close': price['Close']             # days close 
        },
        'volume': {
            '1d' : volume['1d'],
            '1mo': volume['1mo'],
            '1y': volume['1y']
        }
    }
  
    return data



import requests
import yfinance as yf
import pandas as pd
import time
import numpy as np
from fuzzywuzzy import process

def get_ticker_from_name(company_name):
    """
    Queries Yahoo Finance's autocomplete API to find the best matching ticker.
    """
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    
    # Yahoo Finance requires a standard browser User-Agent header to prevent getting blocked
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Grab the first result's symbol
        if data.get('quotes'):
            best_match = data['quotes'][0]
            ticker = best_match['symbol']
            short_name = best_match.get('shortname', company_name)
            
            print(f"Matched '{company_name}' to {ticker} ({short_name})")
            return ticker
    except Exception as e:
        print(f"Error looking up ticker: {e}")
        
    return None

# Test it out

def  get_stock_data(company_name):
    """
    Fetches historical stock data for the given ticker using yfinance.
    """
    ticker = get_ticker_from_name(company_name)
    if not ticker:
        return None

    try:
        stock = yf.Ticker(ticker)
        today = time.strftime("%Y-%m-%d")
        raw_data = yf.download(ticker, start="2020-01-01", end=today)
        raw_data.columns = raw_data.columns.get_level_values(0)
        return raw_data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def engineer_features(
    df: pd.DataFrame,
    vol_windows=(5, 10, 20),
    ma_windows=(5, 10, 20, 50),
    rsi_window: int = 14,
    risk_horizon: int = 5,   # predict volatility over next N days
    return_horizon: int = 1,  # predict return over next N days
) -> pd.DataFrame:
    """
    Build a stationary feature set + targets from raw OHLCV data.
    Assumes df has columns: Open, High, Low, Close, Volume, sorted by date ascending.
    """
   
 
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]
    df_stoclk=df["Close"]
 
    feat = pd.DataFrame(index=df.index)
 
    # ---- A. Returns (core stationary transform) ----
    feat["log_ret_close"] = np.log(close / close.shift(1))
    feat["log_ret_high"] = np.log(high / close.shift(1))
    feat["log_ret_low"] = np.log(low / close.shift(1))
    feat["log_ret_open"] = np.log(open_ / close.shift(1))
    feat["oc_range"] = (close - open_) / open_          # intraday move
    feat["hl_range"] = (high - low) / close              # intraday volatility proxy
 
    # ---- B. Trend: price relative to moving averages (stationary ratio) ----
    for w in ma_windows:
        sma = close.rolling(w).mean()
        feat[f"price_to_sma_{w}"] = close / sma - 1.0
        ema = close.ewm(span=w, adjust=False).mean()
        feat[f"price_to_ema_{w}"] = close / ema - 1.0
 
    # MACD (12/26 EMA difference) + signal line, normalized by price
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_norm"] = macd / close
    feat["macd_signal_norm"] = macd_signal / close
    feat["macd_hist_norm"] = (macd - macd_signal) / close
 
    # ---- C. Momentum ----
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_window).mean()
    avg_loss = loss.rolling(rsi_window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    feat["rsi"] = 100 - (100 / (1 + rs))
    feat["rsi_norm"] = feat["rsi"] / 100.0  # squash to 0-1 for the model
 
    for w in (5, 10, 20):
        feat[f"roc_{w}"] = close.pct_change(w)
 
    # ---- D. Volatility features (also basis for the risk target) ----
    log_ret = feat["log_ret_close"]
    for w in vol_windows:
        feat[f"volatility_{w}"] = log_ret.rolling(w).std()
 
    # ATR (true range accounts for gaps)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean() / close  # normalized by price
 
    # Bollinger Band width
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    feat["bb_width"] = (upper - lower) / sma20
 
    # ---- E. Volume features (relative, not raw) ----
    # log1p first: raw volume is heavily right-skewed (occasional huge spike
    # days) and would otherwise dominate the scaler / swamp other features.
    log_volume = np.log1p(volume)
    log_vol_sma20 = log_volume.rolling(20).mean()
    feat["rel_volume"] = log_volume / log_vol_sma20.replace(0, np.nan)
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    feat["obv_norm"] = obv / obv.rolling(50).std().replace(0, np.nan)
 
    # ---- F. Calendar (optional, low weight features) ----
    feat["day_of_week"] = df.index.dayofweek
    feat["month"] = df.index.month
 
    # ------------------------------------------------------------------
    # TARGETS (computed using FUTURE data on purpose, then we drop the
    # trailing rows where future data doesn't exist -- this is the only
    # place "future" data is allowed to appear)
    # ------------------------------------------------------------------
    # Price target: forward log return over return_horizon days
    feat["target_return"] = np.log(
        close.shift(-return_horizon) / close
    )
 
    # Risk target: realized volatility over the NEXT risk_horizon days
    # (std of daily log returns from t+1 to t+risk_horizon)
    future_rets = log_ret.shift(-1)  # next day's return aligned to today
    feat["target_volatility"] = (
        future_rets.rolling(risk_horizon).std().shift(-(risk_horizon - 1))
    )
    feat.dropna(inplace=True)  # drop rows with NaN values (from rolling windows)
 
    return df_stoclk
def data_collector(company_name):
    stock_data = get_stock_data(company_name)
    if stock_data is not None:
        features = engineer_features(stock_data)

        return features
    else:
        print(f"Failed to collect data for {company_name}.")
        return None
def news_collector(company_name: str, max_articles: int = 20) -> list[str]:
    """
    Returns a list of recent news headlines for the given company
    using yfinance's built-in .news property.
    No API key needed.
    """
    ticker = get_ticker_from_name(company_name)
    tk     = yf.Ticker(ticker)
    news   = tk.news   # list of dicts

    headlines = []
    for article in news[:max_articles]:
        # yfinance news dict structure:
        # { 'title': ..., 'publisher': ..., 'link': ..., 'providerPublishTime': ..., 'type': ... }
        content = article.get("content", {})

        title = (
            content.get("title")             # newer yfinance versions
            or article.get("title")          # older versions
            or ""
        )
        if title:
            headlines.append(title.strip())

    if not headlines:
        print(f"Warning: No news found for {ticker}. Using empty list.")

    print(f"Fetched {len(headlines)} headlines for {ticker}")
    return headlines
if __name__ == "__main__":
    company_name = "Apple Inc."
    data_collector(company_name)

import time
import yfinance as yf
import pandas as pd
import numpy as np

FEATURE_COLS = [
    'returns_lag1', 'returns_lag2', 'returns_lag3', 'returns_lag5',
    'volume_ratio', 'volume_lag1',
    'volatility', 'volatility_20',
    'price_position', 'price_ma50_pos',
    'rsi', 'macd', 'bb_pos', 'hl_range',
]

TECH_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "AMZN"]


def fetch_stock_data(ticker: str, period: str = "1y",
                     retries: int = 3, delay: float = 3.0) -> pd.DataFrame:
    """Fetch OHLCV data — yfinance 1.3.0 handles curl_cffi internally."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # Do NOT pass session — yfinance 1.3.0 manages curl_cffi itself
            stock = yf.Ticker(ticker)
            df    = stock.history(period=period)
            if not df.empty:
                return df
            print(f"[data_loader] Empty response for {ticker} "
                  f"(attempt {attempt}/{retries}), waiting {delay*attempt}s …")
        except Exception as e:
            last_err = e
            print(f"[data_loader] Error fetching {ticker} "
                  f"(attempt {attempt}/{retries}): {e}")
        time.sleep(delay * attempt)

    raise ValueError(
        f"No data for {ticker} after {retries} attempts. "
        f"Last error: {last_err}"
    )


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 14 ML features."""
    df = df.copy()

    # Returns
    df['returns']      = df['Close'].pct_change()
    df['returns_lag1'] = df['returns'].shift(1)
    df['returns_lag2'] = df['returns'].shift(2)
    df['returns_lag3'] = df['returns'].shift(3)
    df['returns_lag5'] = df['returns'].shift(5)

    # Moving averages
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_50'] = df['Close'].rolling(window=50).mean()

    # EMA & MACD
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd']   = df['ema_12'] - df['ema_26']

    # RSI
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_mid']   = df['Close'].rolling(window=20).mean()
    df['bb_std']   = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    band_range     = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_pos']   = (df['Close'] - df['bb_lower']) / band_range

    # Volume
    vol_ma             = df['Volume'].rolling(window=20).mean().replace(0, np.nan)
    df['volume_ratio'] = df['Volume'] / vol_ma
    df['volume_lag1']  = df['volume_ratio'].shift(1)

    # Volatility
    df['volatility']    = df['returns'].rolling(window=10).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()

    # Price vs MAs
    df['price_position'] = (df['Close'] - df['ma_20']) / df['ma_20'].replace(0, np.nan)
    df['price_ma50_pos'] = (df['Close'] - df['ma_50']) / df['ma_50'].replace(0, np.nan)

    # High-Low range
    df['hl_range'] = (df['High'] - df['Low']) / df['Close'].replace(0, np.nan)

    # Target: will next day be positive?
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)

    return df.dropna()


def fetch_market_news(ticker: str = "^GSPC") -> list:
    """Fetch recent news headlines."""
    try:
        obj    = yf.Ticker(ticker)
        news   = obj.news or []
        result = []
        for item in news[:15]:
            content   = item.get("content", {})
            title     = content.get("title", item.get("title", ""))
            summary   = content.get("summary", "")
            pub_at    = content.get("pubDate", "")
            link      = ""
            cp = content.get("canonicalUrl", {})
            if isinstance(cp, dict):
                link = cp.get("url", "")
            publisher = ""
            prov = content.get("provider", {})
            if isinstance(prov, dict):
                publisher = prov.get("displayName", "")
            if title:
                result.append({
                    "title":     title,
                    "summary":   summary,
                    "publisher": publisher,
                    "link":      link,
                    "timestamp": pub_at,
                })
        return result
    except Exception as e:
        print(f"[data_loader] News fetch error: {e}")
        return []


def fetch_ticker_info(ticker: str) -> dict:
    """Return key fundamentals for a ticker."""
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "name":           info.get("longName", ticker),
            "sector":         info.get("sector", "N/A"),
            "market_cap":     info.get("marketCap", 0),
            "pe_ratio":       info.get("trailingPE", 0),
            "52w_high":       info.get("fiftyTwoWeekHigh", 0),
            "52w_low":        info.get("fiftyTwoWeekLow", 0),
            "analyst_target": info.get("targetMeanPrice", 0),
            "recommendation": info.get("recommendationKey", "N/A"),
        }
    except Exception as e:
        print(f"[data_loader] Info fetch error for {ticker}: {e}")
        return {}


def prepare_prediction_data(ticker: str = "AAPL") -> dict:
    """Fetch + engineer all features for ML prediction."""
    df = fetch_stock_data(ticker, period="1y")
    df = add_technical_indicators(df)

    latest_features = df[FEATURE_COLS].iloc[-1:].values  # shape (1, 14)

    return {
        'features':      latest_features,
        'last_price':    float(df['Close'].iloc[-1]),
        'date':          df.index[-1],
        'dataframe':     df,
        'feature_names': FEATURE_COLS,
    }

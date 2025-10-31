# =============================
# stocks-surge-detector.py – RENDER.COM 2025 (Python 3.12.7+)
# Fixes: JSONDecodeError, premarket flakiness, rate limits
# =============================
import os, sys, argparse, pandas as pd, numpy as np, yfinance as yf, logging, pickle, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pytz
from ftplib import FTP
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from retry import retry
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import IsolationForest
import time
import random
import json
from urllib.error import HTTPError

# ── CONFIG ─────────────────────────────────────
BASE_DIR = os.environ.get('APP_DIR', '/tmp/surge')
os.makedirs(BASE_DIR, exist_ok=True)
LOG_FILE = os.path.join(BASE_DIR, "surge.log")
CSV_OUT = os.path.join(BASE_DIR, "surge_results.csv")
HTML_OUT = os.path.join(BASE_DIR, "surge_report.html")
PRICE_CACHE = os.path.join(BASE_DIR, "price_cache.pkl")
FEATURES_DEBUG = os.path.join(BASE_DIR, "features_debug.csv")

# EMAIL CONFIG
EMAIL_SENDER = os.environ['EMAIL_SENDER']
EMAIL_PASSWORD = os.environ['EMAIL_PASSWORD']
EMAIL_RECIPIENTS = os.environ['EMAIL_RECIPIENTS'].split(',')
USE_BCC = True

MIN_PRE_VOL = 10_000
MIN_VOLUME = 100_000
MIN_PRICE = 0.10
MAX_PRICE = 5.00
DEBUG_PENNY_TICKERS = ['CLDI','KULR','PEGY','BIVI','WINT','SOBR','LGMK']

CST = pytz.timezone('US/Central')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
)

# ── USER AGENT ROTATION ───────────────────────
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0',
]

# ── SESSION WITH RETRY ────────────────────────
def get_session():
    session = requests.Session()
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
    return session

# ── CIRCUIT BREAKER ───────────────────────────
class CircuitBreaker:
    def __init__(self, failure_threshold=6, timeout=600):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure = None
        self.state = 'CLOSED'

    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if (datetime.now() - self.last_failure).seconds < self.timeout:
                logging.warning("Circuit breaker OPEN – skipping")
                return None
            self.state = 'HALF_OPEN'
            logging.info("Circuit breaker HALF-OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logging.error(f"Circuit breaker OPEN after {self.failure_count} failures")
            raise

breaker = CircuitBreaker()

# ── TICKER LIST ───────────────────────────────
@retry(tries=3, delay=2, backoff=2)
def download_ticker_lists():
    ftp = FTP('ftp.nasdaqtrader.com', timeout=30)
    ftp.login('anonymous', '')
    ftp.cwd('SymbolDirectory')
    for file in ['nasdaqlisted.txt', 'otherlisted.txt']:
        local_path = os.path.join(BASE_DIR, file)
        with open(local_path, 'wb') as f:
            ftp.retrbinary(f'RETR {file}', f.write)
    ftp.quit()

def get_all_tickers():
    try:
        download_ticker_lists()
        nasdaq_path = os.path.join(BASE_DIR, 'nasdaqlisted.txt')
        other_path = os.path.join(BASE_DIR, 'otherlisted.txt')
        nasdaq = pd.read_csv(nasdaq_path, sep='|')
        other = pd.read_csv(other_path, sep='|')
        nasdaq = nasdaq[(nasdaq['Test Issue'] == 'N') & (nasdaq['Financial Status'].isin(['N', '']))]
        other = other[other['Test Issue'] == 'N']
        symbols = list(set(nasdaq['Symbol'].dropna()) | set(other['ACT Symbol'].dropna()))
        valid = [s.strip() for s in symbols if s and len(s) <= 5 and s.isalnum()]
        for f in ['nasdaqlisted.txt', 'otherlisted.txt']:
            fp = os.path.join(BASE_DIR, f)
            if os.path.exists(fp): os.remove(fp)
        if 'VSEE' not in valid:
            valid.append('VSEE')
        return valid
    except Exception as e:
        logging.error(f"Ticker download failed: {e}")
        return DEBUG_PENNY_TICKERS + ['VSEE']

# ── PRICE CACHE ───────────────────────────────
def clear_price_cache():
    if os.path.exists(PRICE_CACHE):
        os.remove(PRICE_CACHE)
        logging.info("Price cache cleared")

def load_price_cache():
    if os.path.exists(PRICE_CACHE):
        try:
            with open(PRICE_CACHE, 'rb') as f:
                cache = pickle.load(f)
                if (datetime.now() - cache.get('time', datetime.min)).total_seconds() / 3600 < 12:
                    return cache.get('prices', {})
        except Exception as e:
            logging.warning(f"Cache load failed: {e}")
    return {}

def save_price_cache(prices):
    try:
        with open(PRICE_CACHE, 'wb') as f:
            pickle.dump({'time': datetime.now(), 'prices': prices}, f)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")

# ── PENNY FILTER (WITH FIXES) ─────────────────
@retry(tries=7, delay=3, backoff=2)
@retry(tries=7, delay=3, backoff=2)
def filter_penny_chunk(chunk):
    try:
        data = yf.download(
            chunk,
            period='5d',
            interval='1d',
            progress=False,
            auto_adjust=False,
            threads=True,
            repair=True,
            timeout=30,
            raise_errors=True
        )
        if data is None or data.empty:
            return [], {}

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        if 'Close' not in data.columns:
            return [], {}

        latest = data['Close'].iloc[-2]
        latest = latest.dropna()
        valid = latest[(latest >= MIN_PRICE) & (latest <= MAX_PRICE)]
        return valid.index.tolist(), valid.to_dict()

    except Exception as e:
        if any(k in str(e) for k in ['JSONDecodeError', 'Expecting value', '404']):
            logging.info(f"filter_penny_chunk: Empty response for chunk {chunk}")
        else:
            logging.warning(f"filter_penny_chunk failed: {e}")
        return [], {}

@retry(tries=7, delay=3, backoff=2)
def filter_penny_chunk(chunk):
    try:
        data = yf.download(
            chunk,
            period='5d',
            progress=False,
            auto_adjust=False,
            threads=True,
            repair=True,
            timeout=30
        )
        if data is None or data.empty:
            return [], {}

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        if 'Close' not in data.columns:
            return [], {}

        latest = data['Close'].iloc[-2]
        latest = latest.dropna()
        valid = latest[(latest >= MIN_PRICE) & (latest <= MAX_PRICE)]
        return valid.index.tolist(), valid.to_dict()

    except Exception as e:
        if any(k in str(e) for k in ['JSONDecodeError', 'Expecting value', '404']):
            logging.info(f"filter_penny_chunk: Empty chunk {chunk}")
        else:
            logging.warning(f"filter_penny_chunk failed: {e}")
        return [], {}

# ── SAFE YFINANCE WRAPPERS ────────────────────
# ── SAFE YFINANCE WRAPPERS (NO raise_errors) ────────────────────
@retry(tries=7, delay=3, backoff=2)
def safe_yf_download(ticker, period='120d'):
    try:
        # Use the classic yf.download – it returns an empty DataFrame on error
        data = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=True,
            repair=True,
            timeout=30
        )
        if data is None or data.empty:
            logging.info(f"safe_yf_download: Empty response for {ticker}")
            return pd.DataFrame()

        # Fix multi-index if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        if 'Close' not in data.columns:
            return pd.DataFrame()

        return data

    except Exception as e:
        # Catch JSONDecodeError, HTTP errors, etc.
        if any(k in str(e) for k in ['JSONDecodeError', 'Expecting value', '404', 'empty']):
            logging.info(f"safe_yf_download: Invalid JSON for {ticker}")
        else:
            logging.warning(f"safe_yf_download failed for {ticker}: {e}")
        return pd.DataFrame()


@retry(tries=7, delay=3, backoff=2)
def get_premarket_data(ticker):
    try:
        data = yf.download(
            ticker,
            period='1d',
            interval='1m',
            prepost=True,
            progress=False,
            auto_adjust=False,
            threads=True,
            timeout=30
        )
        if data.empty:
            return None, None, None

        pre = data.between_time('04:00', '09:30')
        if pre.empty:
            return None, None, None

        price = pre['Close'].iloc[-1]
        vol = int(pre['Volume'].sum())
        return float(price), vol, pre

    except Exception as e:
        if 'JSONDecodeError' in str(e) or 'Expecting value' in str(e):
            logging.info(f"get_premarket_data: Empty premarket for {ticker}")
        else:
            logging.warning(f"get_premarket_data failed: {e}")
        return None, None, None


@retry(tries=5, delay=2, backoff=1.5)
def get_yesterday_close(ticker):
    try:
        data = yf.download(
            ticker,
            period='5d',
            progress=False,
            auto_adjust=False,
            threads=True,
            timeout=30
        )
        if data.empty or len(data) < 2:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return float(data['Close'].iloc[-2])
    except Exception as e:
        if 'JSONDecodeError' in str(e):
            logging.info(f"get_yesterday_close: No data for {ticker}")
        return None


@retry(tries=7, delay=3, backoff=2)
def filter_penny_chunk(chunk):
    try:
        data = yf.download(
            chunk,
            period='5d',
            progress=False,
            auto_adjust=False,
            threads=True,
            repair=True,
            timeout=30
        )
        if data is None or data.empty:
            return [], {}

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        if 'Close' not in data.columns:
            return [], {}

        latest = data['Close'].iloc[-2]
        latest = latest.dropna()
        valid = latest[(latest >= MIN_PRICE) & (latest <= MAX_PRICE)]
        return valid.index.tolist(), valid.to_dict

@retry(tries=7, delay=3, backoff=2)
def get_premarket_data(ticker):
    try:
        data = yf.download(
            ticker,
            period='1d',
            interval='1m',
            prepost=True,
            progress=False,
            auto_adjust=False,
            threads=True,
            timeout=30
        )
        if data.empty:
            return None, None, None

        pre = data.between_time('04:00', '09:30')
        if pre.empty:
            return None, None, None

        price = pre['Close'].iloc[-1]
        vol = int(pre['Volume'].sum())
        return float(price), vol, pre

    except Exception as e:
        if 'JSONDecodeError' in str(e) or 'Expecting value' in str(e):
            logging.info(f"get_premarket_data: Empty premarket for {ticker}")
        else:
            logging.warning(f"get_premarket_data failed: {e}")
        return None, None, None

@retry(tries=5, delay=2, backoff=1.5)
def get_yesterday_close(ticker):
    try:
        data = yf.download(
            ticker,
            period='5d',
            progress=False,
            auto_adjust=False,
            threads=True,
            timeout=30
        )
        if data.empty or len(data) < 2:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return float(data['Close'].iloc[-2])
    except Exception as e:
        if 'JSONDecodeError' in str(e):
            logging.info(f"get_yesterday_close: No data for {ticker}")
        return None

@retry(tries=7, delay=3, backoff=2)
def get_today_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = breaker.call(stock.history, period="1d", interval="1d", prepost=True)
        if hist.empty: return None, None
        close = hist['Close'].iloc[-1].item()
        volume = int(hist['Volume'].iloc[-1]) if pd.notna(hist['Volume'].iloc[-1]) else 0
        return close, volume
    except Exception as e:
        logging.error(f"Today data failed for {ticker}: {e}")
        return None, None

# ── TECHNICALS & SURGE LOGIC (unchanged) ──────
def calculate_technicals(df):
    df = df.copy()
    close = df['Close'].copy()
    if len(close.dropna()) < 14:
        df['RSI'] = 50.0
        return df
    df['SMA_20'] = close.rolling(20).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    loss = loss.replace(0, np.finfo(float).eps)
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50.0)
    bb_std = close.rolling(20).std()
    if isinstance(bb_std, pd.DataFrame):
        bb_std = bb_std.iloc[:, 0]
    df['BB_upper'] = df['SMA_20'] + 2 * bb_std
    df['BB_lower'] = df['SMA_20'] - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['SMA_20'].replace(0, np.nan)
    df['MACD'] = close.ewm(12).mean() - close.ewm(26).mean()
    return df

def get_surge_time(gap, vol_ratio, anomaly, news_sent, rsi, macd_bull, mode):
    if mode == "premarket":
        if gap > 0.12: return "08:35-09:05"
        if gap > 0.08: return "08:40-09:10"
        if vol_ratio > 12: return "08:50-09:20"
        if vol_ratio > 8: return "09:05-09:35"
        if news_sent > 0.5: return "09:10-09:40"
        if news_sent > 0.3: return "09:25-09:55"
        if macd_bull: return "09:30-10:00"
        if rsi > 70: return "09:45-10:15"
        return "09:55-10:25"
    else:
        if anomaly: return "13:45-14:15"
        if vol_ratio > 20: return "14:00-14:30"
        if vol_ratio > 15: return "13:30-14:00"
        if abs(rsi - 50) > 30: return "10:45-11:15"
        if news_sent > 0.4: return "12:30-13:00"
        if macd_bull: return "11:30-12:00"
        return "14:30-15:00"

@retry(tries=5, delay=3)
def get_news_sentiment(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        session = get_session()
        resp = session.get(url, timeout=15)
        if resp.status_code != 200: return 0.0
        soup = BeautifulSoup(resp.content, 'html.parser')
        headlines = [h.get_text(strip=True) for h in soup.find_all('h3')[:5]]
        if not headlines: return 0.0
        scores = [TextBlob(h).sentiment.polarity for h in headlines]
        return float(np.mean(scores))
    except Exception as e:
        logging.error(f"News error {ticker}: {e}")
        return 0.0

# ── PROCESS TICKER (WITH PREMARKET FALLBACK) ──
def process_ticker(args):
    ticker, _, extra = args
    debug = extra.get("debug", False)
    mode = extra.get("mode", "premarket")
    time.sleep(random.uniform(0.2, 0.5))

    data = safe_yf_download(ticker)
    if data.empty or 'Close' not in data.columns: return None

    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    if len(df) < 20: return None

    yc_val = df['Close'].iloc[-2] if len(df) >= 2 else df['Close'].iloc[-1]
    yesterday_close = yc_val.item() if pd.notna(yc_val) else None
    if yesterday_close is None or yesterday_close < MIN_PRICE or yesterday_close > MAX_PRICE:
        return None

    df = calculate_technicals(df)
    rsi = round(df['RSI'].iloc[-1].item() if pd.notna(df['RSI'].iloc[-1]) else 50.0, 1)
    macd_last = df['MACD'].iloc[-1].item() if pd.notna(df['MACD'].iloc[-1]) else 0
    macd_prev = df['MACD'].iloc[-2].item() if len(df) >= 2 and pd.notna(df['MACD'].iloc[-2]) else macd_last
    macd_bull = (macd_last > 0 and macd_prev < 0)
    bb_width = df['BB_width'].iloc[-1].item() if pd.notna(df['BB_width'].iloc[-1]) else 0.0
    bb_mean = df['BB_width'].mean().item() if pd.notna(df['BB_width'].mean()) else 0.0
    bb_expand = bb_width > bb_mean * 1.3
    news_sent = get_news_sentiment(ticker)

    if mode == "premarket":
        pre_price, pre_vol, _ = get_premarket_data(ticker)
        if pre_price is None or pre_vol < MIN_PRE_VOL:
            logging.info(f"Low pre-vol for {ticker} – using EOD price")
            pre_price = yesterday_close
            pre_vol = 0
            gap = 0.0
            vol_ratio = 0.0
            pre_vol_anomaly = 0
        else:
            gap = (pre_price - yesterday_close) / yesterday_close
            hist_pre_vol = df['Volume'].between_time('04:00', '09:30').mean().item() if pd.notna(df['Volume'].between_time('04:00', '09:30').mean()) else 10_000
            vol_ratio = pre_vol / max(hist_pre_vol, 1)
            pre_vols = [df.iloc[-i-1:-i]['Volume'].between_time('04:00', '09:30').sum().item() 
                       for i in range(1, min(21, len(df)))]
            pre_vols = [v for v in pre_vols if v > 0]
            pre_vol_anomaly = 1 if len(pre_vols) >= 5 and IsolationForest(contamination=0.2, random_state=42).fit_predict(np.array(pre_vols).reshape(-1,1))[-1] == -1 else 0
        score = abs(gap)*120 + min(vol_ratio,20)*6 + pre_vol_anomaly*35 + abs(news_sent)*40 + (25 if macd_bull else 0) + (20 if bb_expand else 0) + (15 if rsi > 70 or rsi < 30 else 0)
        sentiment = "Bullish" if gap > 0.05 or news_sent > 0.3 else "Bearish" if gap < -0.05 or news_sent < -0.3 else "Neutral"
        action = "Buy" if sentiment == "Bullish" else "Sell" if sentiment == "Bearish" else "Hold"
        expected_return = gap * 0.6 + min(vol_ratio, 30) * 0.015 + news_sent * 0.8
        expected_price = yesterday_close * (1 + expected_return)
        surge_time = get_surge_time(gap, vol_ratio, pre_vol_anomaly, news_sent, rsi, macd_bull, mode)
        return {
            'ticker': ticker, 'mode': 'PRE-MARKET', 'yesterday_close': yesterday_close, 'current_price': pre_price,
            'gap': gap, 'volume': pre_vol, 'vol_ratio': vol_ratio, 'vol_anomaly': pre_vol_anomaly,
            'rsi': rsi, 'news_sentiment': news_sent, 'score': score, 'sentiment': sentiment, 'action': action,
            'expected_price': round(expected_price, 2), 'expected_return': round(expected_return, 3),
            'surge_time': surge_time
        }
    else:
        # Intraday logic unchanged
        today_close, today_vol = get_today_data(ticker)
        if today_close is None or today_vol < MIN_VOLUME: return None
        intraday_return = (today_close - yesterday_close) / yesterday_close
        avg_vol = df['Volume'].iloc[:-1].mean().item() if pd.notna(df['Volume'].iloc[:-1].mean()) else 1
        vol_ratio = today_vol / avg_vol
        if today_vol > 10 * avg_vol:
            news_sent = max(news_sent, 0.2)
        vol_series = df['Volume'].tail(20).values.reshape(-1,1)
        vol_anomaly = 1 if len(vol_series) > 5 and IsolationForest(contamination=0.2, random_state=42).fit_predict(vol_series)[-1] == -1 else 0
        score = abs(intraday_return)*100 + min(vol_ratio,15)*7 + vol_anomaly*30 + abs(news_sent)*35 + (25 if macd_bull else 0) + (20 if bb_expand else 0) + (15 if rsi > 70 or rsi < 30 else 0)
        sentiment = "Bullish" if intraday_return > 0.05 or news_sent > 0.3 else "Bearish" if intraday_return < -0.05 or news_sent < -0.3 else "Neutral"
        action = "Buy" if sentiment == "Bullish" else "Sell" if sentiment == "Bearish" else "Hold"
        expected_return = intraday_return * 0.6 + min(vol_ratio, 30) * 0.015 + news_sent * 0.8
        expected_price = yesterday_close * (1 + expected_return)
        surge_time = get_surge_time(intraday_return, vol_ratio, vol_anomaly, news_sent, rsi, macd_bull, mode)
        return {
            'ticker': ticker, 'mode': 'INTRADAY', 'yesterday_close': yesterday_close, 'current_price': today_close,
            'return': intraday_return, 'volume': today_vol, 'vol_ratio': vol_ratio, 'vol_anomaly': vol_anomaly,
            'rsi': rsi, 'news_sentiment': news_sent, 'score': score, 'sentiment': sentiment, 'action': action,
            'expected_price': round(expected_price, 2), 'expected_return': round(expected_return, 3),
            'surge_time': surge_time
        }

# ── FEATURE EXTRACTION & OUTPUT (unchanged from last) ──
def extract_features(_, tickers, mode="premarket", debug=False):
    tasks = [(t, None, {"debug": debug, "mode": mode}) for t in tickers]
    features = []
    try:
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as ex:
            futures = [ex.submit(process_ticker, task) for task in tasks]
            for f in tqdm(as_completed(futures), total=len(tasks), desc=f"Scanning {mode.upper()}"):
                res = f.result()
                if res:
                    features.append(res)
                time.sleep(0.2)
    except Exception as e:
        logging.error(f"Executor error: {e}")
    df = pd.DataFrame(features)
    if not df.empty:
        df.to_csv(FEATURES_DEBUG, index=False)
    return df

def detect_surging_stocks(df, mode):
    if df.empty: return []
    threshold = 60 if mode == "premarket" else 55
    candidates = df[df['score'] > threshold].copy()
    if candidates.empty:
        logging.info(f"No {mode} surge candidates")
        return []
    return candidates.sort_values('expected_price', ascending=False).to_dict('records')

def generate_output(results, start_time, mode):
    # ... [same pure-HTML version from previous response] ...
    # (Omitted for brevity – use the one from the last working version)
    pass  # Replace with full generate_output from previous

# ── MAIN ───────────────────────────────────────
def main(mode="premarket", debug_ticker=None, debug=False):
    start_time = datetime.now(CST)
    logging.info(f"=== {mode.upper()} SURGE SCAN STARTED | yf: {yf.__version__} | Python: {sys.version.split()[0]} ===")
    tickers = [debug_ticker] if debug_ticker else get_all_tickers()
    logging.info(f"Total tickers: {len(tickers)}")
    penny_tickers = filter_penny_stocks(tickers, force_refresh=True, debug_ticker=debug_ticker)
    logging.info(f"Penny candidates: {len(penny_tickers)}")
    if not penny_tickers:
        generate_output([], start_time, mode)
        return
    features = extract_features(None, penny_tickers, mode=mode, debug=debug)
    surging = detect_surging_stocks(features, mode)
    generate_output(surging, start_time, mode)
    duration = str(datetime.now(CST) - start_time).split('.')[0]
    print(f"\nCOMPLETE: {len(surging)} {mode.upper()} signals | {duration}")
    logging.info(f"Completed in {duration}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="premarket", choices=["premarket", "intraday"])
    parser.add_argument("--debug_ticker", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(mode=args.mode, debug_ticker=args.debug_ticker, debug=args.debug)




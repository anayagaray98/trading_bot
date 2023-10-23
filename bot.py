import ccxt
import config
import schedule
import pandas as pd
import numpy as np
from datetime import datetime
import time
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')
from .utils import get_ema, get_macd, stochrsi


""" Configuration Variables """

pair = 'DOTUP/USDT'
exchange = ccxt.binance({
    "apiKey": config.API_KEY,
    "secret": config.API_SECRET
})


def run_bot():
    
    print(f"Fetching new bars for {datetime.now().isoformat()}")
    bars = exchange.fetch_ohlcv(pair, timeframe='15m', limit=100)
    df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    stochrsi(df)
    get_macd(df)
    get_ema(df)
    
    print(df)


schedule.every(5).seconds.do(run_bot)
while True:
    schedule.run_pending()
    time.sleep(1)
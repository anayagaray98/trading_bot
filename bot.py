import ccxt
import config
import pandas as pd
from datetime import datetime
import warnings
from utils import calculate_ema, calculate_macd, calculate_stochrsi, calculate_adx, \
    calculate_obv, calculate_chaikin_oscillator, calculate_pivot_points, \
        calculate_price_channels, calculate_mass_index, calculate_elliott_wave, calculate_williams_percent_r, \
            calculate_bollinger_bands, calculate_ichimoku_cloud, calculate_atr, calculate_stoch

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)

#____________________________________________________________________________________________________


""" Configuration Variables """

pairs = ['ETH/USDT'] #'SOL/USDT', 'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT'
candle_type = '1m' # Since we're trading on the Futures market with leverage
history_limit = 1500 # This is the largest size per API call.
exchange = ccxt.binance({
    "apiKey": config.API_KEY_PRODUCTION,
    "secret": config.API_SECRET_PRODUCTION,
    'options': {
        'defaultType': 'future',
    },
})

# exchange.set_sandbox_mode(True)  # comment if you're not using the testnet
markets = exchange.load_markets()
exchange.verbose = False  # debug output

#____________________________________________________________________________________________________________

def get_account_positions():
    balance = exchange.fetch_balance()
    positions = balance['info']['positions']
    return positions

def set_leverage(leverage, pair):
    exchange.set_leverage(leverage, pair)

def place_order(symbol, quantity, side, price, order_type):
    try:
        print(f"Placing {side} order for {quantity} {symbol} at price {price}")
        order = exchange.create_order(symbol, order_type, side, quantity, price)
        print("Order details:")
        print(order)
    except Exception as e:
        print(f"An error occurred while placing the order: {e}")

def run_bot():
    for pair in pairs:
        print(f"Fetching new bars for {datetime.now().isoformat()}")
        bars = exchange.fetch_ohlcv(pair, timeframe=candle_type, limit=history_limit)
        df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        indicator_functions = [
            calculate_stochrsi,
            calculate_macd,
            calculate_ema,
            calculate_adx,
            calculate_chaikin_oscillator,
            calculate_pivot_points,
            calculate_price_channels,
            calculate_mass_index,
            calculate_elliott_wave,
            calculate_williams_percent_r,
            calculate_obv,
            calculate_bollinger_bands,
            calculate_ichimoku_cloud,
            calculate_atr,
            calculate_stoch
        ]

        # Apply each indicator function to the DataFrame
        for indicator_function in indicator_functions:
            df = indicator_function(df)

        
        # Function to dynamically calculate threshold values for Bollinger Bands
        def calculate_bollinger_bands_thresholds(df, periods=5):
            # Calculate the rolling average of the last 'periods' periods for bollinger_upper and bollinger_lower
            df['rolling_upper_avg'] = df['bollinger_upper'].rolling(periods).mean()
            df['rolling_lower_avg'] = df['bollinger_lower'].rolling(periods).mean()

            # Take the last calculated average values
            upper_threshold = df['rolling_upper_avg'].iloc[-1]
            lower_threshold = df['rolling_lower_avg'].iloc[-1]

            return (lower_threshold, upper_threshold)
        

        # Function to dynamically calculate Ichimoku Cloud threshold values
        def calculate_ichimoku_cloud_thresholds(df, moving_average_period=20):

            # Calculate moving averages of Senkou Span A and Senkou Span B
            df['senkou_span_a_ma'] = df['senkou_span_a'].rolling(window=moving_average_period).mean()
            df['senkou_span_b_ma'] = df['senkou_span_b'].rolling(window=moving_average_period).mean()

            # Set the threshold values for Ichimoku Cloud based on the moving averages
            ichimoku_threshold = (df['senkou_span_b_ma'].iloc[-1], df['senkou_span_a_ma'].iloc[-1])

            return ichimoku_threshold
        
        # Function to dynamically calculate MACD threshold values for upper and lower bounds
        def calculate_macd_thresholds(df, moving_average_period=20, lower_threshold_percentage=0.8):
            # Calculate the MACD threshold based on a 20-period moving average of the crossover points
            df['macd_crossover'] = (df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))
            df['macd_crossover_ma'] = df['macd_crossover'].rolling(window=moving_average_period).mean()
            upper_threshold = df['macd_crossover_ma'].shift(1).iloc[-1]  # Use the last value for upper threshold

            # Calculate the lower threshold as a percentage of the upper threshold
            lower_threshold = lower_threshold_percentage * upper_threshold

            return (lower_threshold, upper_threshold)
        
        # Function to dynamically calculate OBV threshold values
        def calculate_obv_thresholds(df, moving_average_period=20):

            # Calculate the OBV threshold based on a moving average
            df['obv_threshold'] = df['obv'].rolling(window=moving_average_period).mean()

            return (0, df['obv_threshold'].iloc[-1])


        bollinger_thresholds = calculate_bollinger_bands_thresholds(df, periods=5)
        ichimoku_thresholds = calculate_ichimoku_cloud_thresholds(df)
        macd_thresholds = calculate_macd_thresholds(df)
        obv_thresholds = calculate_obv_thresholds(df)

        # technical indicators to use
        indicators = [
            'rsi',
            'macd',
            'chaikin_oscillator',
            'bollinger_bands',
            'atr',
            'stoch',
            'ichimoku_cloud',
            'williams_percent_r',
            'adx',
            'obv',
        ]

        # Define threshold values for each indicator
        thresholds = {
            'rsi': (30, 70), # RSI threshold values
            'macd': macd_thresholds, # MACD threshold values
            'chaikin_oscillator': (-0.2, 0.2), # Chaikin Oscillator threshold values
            'bollinger_bands': bollinger_thresholds, # Bollinger Bands threshold values
            'atr': (14, 35), # ATR threshold values
            'stoch': (20, 80), # Stochastic Oscillator threshold values
            'ichimoku_cloud': ichimoku_thresholds, # Ichimoku Cloud threshold values
            'williams_percent_r': (20, 80), # Williams %R threshold values
            'adx': (25, 50), # ADX threshold values
            'obv': obv_thresholds, # On-Balance Volume threshold values
        }

        # Initialize buy and sell signals as 0 (Hold)
        df['buy_signal'] = 0
        df['buy_signal_confidence'] = 0.0
        df['sell_signal'] = 0
        df['sell_signal_confidence'] = 0.0

        # Loop through the DataFrame to calculate buy and sell signals and confidence levels
        for i in range(len(df)):
            bullish_indicators = 0
            bearish_indicators = 0

            for indicator in indicators:

                buy_threshold, sell_threshold = thresholds[indicator]

                if indicator == 'bollinger_bands':
                    # Check if the upper Bollinger Band crosses the upper threshold
                    if df['bollinger_upper'][i] > buy_threshold:
                        bullish_indicators += 1
                    # Check if the lower Bollinger Band crosses the lower threshold
                    elif df['bollinger_lower'][i] < sell_threshold:
                        bearish_indicators += 1

                elif indicator == 'ichimoku_cloud':
                    # Check if Senkou Span A crosses above Senkou Span B (upper_threshold) for bullish
                    if df['senkou_span_a'][i] > buy_threshold:
                        bullish_indicators += 1
                    # Check if Senkou Span A crosses below Senkou Span B (lower_threshold) for bearish
                    elif df['senkou_span_b'][i] < sell_threshold:
                        bearish_indicators += 1

                elif indicator == 'stoch':
                    if df['stoch_k'][i] > df['stoch_d'][i] > buy_threshold:
                        bullish_indicators += 1
                    elif df['stoch_k'][i] < df['stoch_d'][i] < sell_threshold:
                        bearish_indicators += 1

                else:
                    if df[indicator][i] > buy_threshold:
                        bullish_indicators += 1
                    elif df[indicator][i] < sell_threshold:
                        bearish_indicators += 1

            # Calculate buy signal and confidence level
            if bullish_indicators > bearish_indicators:
                df.at[i, 'buy_signal'] = 1
                df.at[i, 'buy_signal_confidence'] = bullish_indicators / len(indicators)
            else:
                df.at[i, 'buy_signal'] = 0
                df.at[i, 'buy_signal_confidence'] = 0.0  # No buy signal

            # Calculate sell signal and confidence level
            if bearish_indicators > bullish_indicators:
                df.at[i, 'sell_signal'] = 1
                df.at[i, 'sell_signal_confidence'] = bearish_indicators / len(indicators)
            else:
                df.at[i, 'sell_signal'] = 0
                df.at[i, 'sell_signal_confidence'] = 0.0  # No sell signal


        df.to_csv(f"data/trading_info_for_model_{pair.replace('/', '_')}.csv")

        # # Get the trading signal from the model
        # signal = get_model_signal(best_model, X)

        # print(f"Offer: {signal}")

        # TRADING PART RIGHT HERE
        # Example: Place a limit order to buy 1 unit of the asset if the model predicts a buy signal

        # Calculate the target price based on the model's predictions
        # current_price = df['close'].iloc[-1]
        # if signal == 1:
        #     # If the model predicts a buy signal, you may consider a target price higher than the current price.
        #     target_price = current_price * 1.05  # You can adjust the multiplier as needed.
        #     place_order(pair, 1, 'buy', target_price)
        # elif signal == -1:
        #     # If the model predicts a sell signal, you may consider a target price lower than the current price.
        #     target_price = current_price * 0.95  # You can adjust the multiplier as needed.
        #     place_order(pair, 1, 'sell', target_price)

run_bot()
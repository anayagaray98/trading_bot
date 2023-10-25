import ccxt
import config
import pandas as pd
from datetime import datetime
import schedule
import warnings
import time
from utils import calculate_ema, calculate_macd, calculate_stochrsi, calculate_adx, \
    calculate_obv, calculate_chaikin_oscillator, calculate_pivot_points, \
        calculate_price_channels, calculate_mass_index, calculate_elliott_wave, calculate_williams_percent_r, \
            calculate_bollinger_bands, calculate_ichimoku_cloud, calculate_atr, calculate_stoch

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)

#____________________________________________________________________________________________________


""" Configuration Variables """

pairs = ['DOGE/USDT'] #'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'MATIC/USDT', 'FLM/USDT', 'REEF/USDT', 'DOGE/USDT', 'XRP/USDT', 'GALA/USDT'
candle_types = ['1m', '5m'] # Since we're trading on the Futures market with leverage
history_limit = 1500 # This is the largest size per API call.
allowed_confidence_threshold = 0.65 # This is the minimum confidence level to make a buy/sell decision.
trade_quantity_amount = 20.00 # Quantity in USDT
leverage = 7
type_of_order = 'market' # limit, market
expected_return_level = 6 # %
stop_loss_level = 3 # %

exchange = ccxt.binance({
    "apiKey": config.API_KEY_PRODUCTION,
    "secret": config.API_SECRET_PRODUCTION,
    'options': {
        'defaultType': 'future',
    },
})

exchange.verbose = False  # debug output

#____________________________________________________________________________________________________________

def get_account_positions(pair):
    balance = exchange.fetch_balance()
    positions = balance['info']['positions']
    pair = pair.replace('/', '')
    matching_positions = []
    for position in positions:
        if position['symbol'] == pair:
            matching_positions.append(position)
    return matching_positions
            
def set_leverage(leverage, pair):
    pair = pair.replace('/', '')
    exchange.set_leverage(leverage, pair)

def place_order(symbol, quantity, side, price, order_type, params):
    try:
        print(f"Placing {side} order for {quantity} {symbol} at price {price}")
        if order_type == 'limit':
            order = exchange.create_order(symbol=symbol, type=order_type, side=side, amount=quantity, price=price, params=params)
        elif order_type == 'market':
            order = exchange.create_order(symbol=symbol, type=order_type, side=side, amount=quantity, params=params)
        print("Order details:")
        print(order)
    except Exception as e:
        print(f"An error occurred while placing the order: {e}")

def run_bot():
    for pair in pairs:
        print(f"Fetching new bars for {datetime.now().isoformat()}")

        buy_signals = []
        sell_signals = []
        buy_confidences = []
        sell_confidences = []
        current_price = 0.0

        for candle_type in candle_types:

            try:
                bars = exchange.fetch_ohlcv(pair, timeframe=candle_type, limit=history_limit)
            except Exception as e:
                print(f"An error occured: {e}")
                time.sleep(5)
                pass

            df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            if candle_type == '1m':
                current_price += float(df['close'].iloc[-1])

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

            print("*"*80)
            print(f"SUMMARY: {candle_type} candles.")
            print("*"*80)
            print(df[['buy_signal', 'buy_signal_confidence', 'sell_signal', 'sell_signal_confidence']].tail(10))

            # Get the last 5 rows of the DataFrame
            last_5_rows = df.tail(5)

            # Extract the last row for 'buy_signal' and 'sell_signal'
            last_row = last_5_rows.iloc[-1]

            buy_signals.append(last_row['buy_signal'])
            sell_signals.append(last_row['sell_signal'])

            # Calculate the mean confidence for the last 5 rows
            buy_confidences.append(last_5_rows['buy_signal_confidence'].mean())
            sell_confidences.append(last_5_rows['sell_signal_confidence'].mean())

        signals = []
        for i in range(len(candle_types)):
            if buy_signals[i] == 1 and sell_signals[i] == 0 and buy_confidences[i] > allowed_confidence_threshold:
                signals.append('buy')
            elif buy_signals[i] == 1 and sell_signals[i] == 1:
                if buy_confidences[i] > allowed_confidence_threshold:
                    signals.append('buy')
                elif sell_confidences[i] > allowed_confidence_threshold:
                    signals.append('sell')
            elif sell_signals[i] == 1 and sell_confidences[i] > allowed_confidence_threshold:
                signals.append('sell')

        set_leverage(leverage, pair)
        opened_positions = get_account_positions(pair)
        # print(opened_positions)

        in_position = False
        set_position_side = None
        set_position_amount = 0.0
        has_notional = False
        set_entry_price = 0.0

        for position in opened_positions:
            position_amt = float(position['positionAmt'])
            position_side = position['positionSide']
            bid_notional = float(position['bidNotional'])
            ask_notional = float(position['askNotional'])
            entry_price = float(position['entryPrice'])

            if bid_notional > 0.0 or ask_notional > 0.0:
                has_notional = True

            if position_side == 'LONG' and position_amt > 0 and (bid_notional == 0.0 or ask_notional == 0.0):
                in_position = True
                set_position_side = 'LONG'
                set_position_amount += position_amt
                set_entry_price += entry_price
                break
            elif position_side == 'SHORT' and position_amt < 0 and (bid_notional == 0.0 or ask_notional == 0.0):
                in_position = True
                set_position_side = 'SHORT'
                set_position_amount += position_amt
                set_entry_price += entry_price
                break
        
        if all(signal == 'buy' for signal in signals):
            final_signal = 'buy'
        elif all(signal == 'sell' for signal in signals):
            final_signal = 'sell'
        else:
            final_signal = None

        if final_signal:
            if set_position_side:
                print(f"Open Order: {set_position_side}")
            else:
                print("There is no opened position.")            

            # TRADING PART RIGHT HERE
            if not in_position and not has_notional:
                print(f"Trying to create an offer: {final_signal}")
                if final_signal == 'buy':
                    params = {'positionSide': 'LONG', 'leverage':leverage}
                    new_quantity = (trade_quantity_amount * leverage) / current_price
                    target_price = current_price * 0.99
                    place_order(symbol=pair, quantity=new_quantity, side='buy', price=target_price, order_type=type_of_order, params=params)
                else:
                    params = {'positionSide': 'SHORT', 'leverage':leverage}
                    new_quantity = (trade_quantity_amount * leverage) / current_price
                    target_price = current_price * 1.01
                    place_order(symbol=pair, quantity=new_quantity, side='sell', price=target_price, order_type=type_of_order, params=params)
            
            # If there is opened position
            else:
                # This would trigger the stop loss and take profit
                price_change = ((current_price/set_entry_price)-1)*100
                print(f"CURRENT PRICE CHANGE: {round(price_change, 2)} %")

                if set_position_side == 'SHORT':

                    if (price_change <= (expected_return_level * -1)):
                        print("Triggering TAKE PROFIT")
                    if (price_change > stop_loss_level):
                        print("Triggering STOP LOSS")

                    if (price_change <= (expected_return_level * -1)) or (price_change > stop_loss_level):
                        params = {'positionSide': 'SHORT', 'leverage':leverage}
                        place_order(symbol=pair, quantity=set_position_amount*-1, side='buy', price=current_price, order_type=type_of_order, params=params)
                    
                    else:
                        # If stop loss and take profit are not triggered, then it continues to the normla flow
                        if final_signal == 'buy':
                            if set_position_side == 'SHORT':
                                print(f"Trying to cancel an offer: {set_position_side}")
                                params = {'positionSide': 'SHORT', 'leverage':leverage}
                                target_price = current_price * 0.99
                                place_order(symbol=pair, quantity=set_position_amount*-1, side='buy', price=target_price, order_type=type_of_order, params=params)
                        else:
                            if set_position_side == 'LONG':
                                print(f"Trying to cancel an offer: {set_position_side}")
                                params = {'positionSide': 'LONG', 'leverage':leverage}
                                target_price = current_price * 1.01
                                place_order(symbol=pair, quantity=set_position_amount*-1, side='sell', price=target_price, order_type=type_of_order, params=params)

                if set_position_side == 'LONG':

                    if (price_change >= expected_return_level):
                        print("Triggering TAKE PROFIT")
                    if (price_change < (stop_loss_level * -1)):
                        print("Triggering STOP LOSS")

                    if (price_change >= expected_return_level) or (price_change < (stop_loss_level * -1)):
                        params = {'positionSide': 'LONG', 'leverage':leverage}
                        place_order(symbol=pair, quantity=set_position_amount*-1, side='sell', price=current_price, order_type=type_of_order, params=params)

                    else:
                        # If stop loss and take profit are not triggered, then it continues to the normla flow
                        if final_signal == 'buy':
                            if set_position_side == 'SHORT':
                                print(f"Trying to cancel an offer: {set_position_side}")
                                params = {'positionSide': 'SHORT', 'leverage':leverage}
                                target_price = current_price * 0.99
                                place_order(symbol=pair, quantity=set_position_amount*-1, side='buy', price=target_price, order_type=type_of_order, params=params)
                        else:
                            if set_position_side == 'LONG':
                                print(f"Trying to cancel an offer: {set_position_side}")
                                params = {'positionSide': 'LONG', 'leverage':leverage}
                                target_price = current_price * 1.01
                                place_order(symbol=pair, quantity=set_position_amount*-1, side='sell', price=target_price, order_type=type_of_order, params=params)

        else:
            print("Nothing to do. There is no significant signal here.")

schedule.every(30).seconds.do(run_bot)

while True:
    schedule.run_pending()
    time.sleep(5)
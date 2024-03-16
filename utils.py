import pandas as pd
import numpy as np

def calculate_stochrsi(df, period=14, smoothK=3, smoothD=3):
    """
    Calculate Stochastic RSI (StochRSI) with optional smoothing.
    
    Args:
    df (pandas.DataFrame): DataFrame with 'close' prices.
    period (int): StochRSI period (default is 14).
    smoothK (int): Smoothing period for %K (default is 3).
    smoothD (int): Smoothing period for %D (default is 3).
    
    Returns:
    pandas.DataFrame: Original DataFrame with 'rsi', 'srsi_k', and 'srsi_d' columns added.
    """
    
    # Calculate RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate StochRSI
    stochrsi = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
    stochrsi_k = stochrsi.rolling(window=smoothK).mean() * 100
    stochrsi_d = stochrsi_k.rolling(window=smoothD).mean()

    df['rsi'] = rsi
    df['srsi_k'] = stochrsi_k
    df['srsi_d'] = stochrsi_d

    return df

def calculate_stoch(df, period=14, smoothK=3, smoothD=3):
    """
    Calculate Stochastic Oscillator (Stoch) with optional smoothing.
    
    Args:
    df (pandas.DataFrame): DataFrame with 'close' prices.
    period (int): Stochastic period (default is 14).
    smoothK (int): Smoothing period for %K (default is 3).
    smoothD (int): Smoothing period for %D (default is 3).
    
    Returns:
    pandas.DataFrame: Original DataFrame with 'stoch_k' and 'stoch_d' columns added.
    """
    
    # Calculate the lowest low and highest high over the period
    l14 = df['close'].rolling(period).min()
    h14 = df['close'].rolling(period).max()
    
    # Calculate %K with smoothing
    df['stoch_k'] = ((df['close'].shift(1) - l14) * 100) / (h14 - l14)
    df['stoch_k'] = df['stoch_k'].rolling(smoothK).mean()  # Smooth %K
    
    # Calculate %D by smoothing %K
    df['stoch_d'] = df['stoch_k'].rolling(smoothD).mean()
    
    return df

def calculate_macd(df, slow = 26, fast = 12, smooth = 9):
    """
    Calculate Moving Average Convergence Divergence (MACD) with signal and histogram.

    Args:
    df (pandas.DataFrame): DataFrame with 'close' prices.

    Returns:
    pandas.DataFrame: Original DataFrame with 'macd', 'signal', and 'hist' columns added.
    """

    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = pd.DataFrame(exp1 - exp2)
    df['signal'] = pd.DataFrame(df['macd'].ewm(span=smooth, adjust=False).mean())
    df['hist'] = pd.DataFrame(df['macd'] - df['signal'])

    return df

def calculate_ema(df):
    """
    Calculate Exponential Moving Averages (EMA) for 50 and 200 periods.

    Args:
    df (pandas.DataFrame): DataFrame with 'close' prices.

    Returns:
    pandas.DataFrame: Original DataFrame with 'SMA50', 'SMA200', 'EMA50', and 'EMA200' columns added.
    """
    ema50_window = 50
    ema200_window = 200
    ema50_multiplier = np.round((2 / (ema50_window + 1)), 2)
    ema200_multiplier = np.round((2 / (ema200_window + 1)), 2)

    df['SMA50'] = np.round(df['close'].rolling(window=50, center=False).mean(), 4)
    df['SMA200'] = np.round(df['close'].rolling(window=200, center=False).mean(), 4)
    df['EMA50'] = df['SMA50']
    df['EMA200'] = df['SMA200']
    df['EMA50'] = (df['close'] * ema50_multiplier) + (df['EMA50'].shift(1) * (1 - ema50_multiplier))
    df['EMA200'] = (df['close'] * ema200_multiplier) + (df['EMA200'].shift(1) * (1 - ema200_multiplier))

    return df

def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    """
    Calculate Bollinger Bands.

    Args:
    df (pandas.DataFrame): DataFrame with 'close' prices.
    window (int): Window for calculating the moving average (default is 20).
    num_std_dev (int): Number of standard deviations for the bands (default is 2).

    Returns:
    pandas.DataFrame: Original DataFrame with 'bollinger_upper', 'bollinger_middle', and 'bollinger_lower' columns added.
    """
    df['bollinger_middle'] = df['close'].rolling(window=window).mean()
    df['bollinger_std'] = df['close'].rolling(window=window).std()
    df['bollinger_upper'] = df['bollinger_middle'] + (num_std_dev * df['bollinger_std'])
    df['bollinger_lower'] = df['bollinger_middle'] - (num_std_dev * df['bollinger_std'])
    return df

def calculate_atr(df, window=14):
    """
    Calculate Average True Range (ATR).

    Args:
    df (pandas.DataFrame): DataFrame with 'high', 'low', and 'close' prices.
    window (int): Window for calculating ATR (default is 14).

    Returns:
    pandas.DataFrame: Original DataFrame with 'atr' column added.
    """
    df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    df['atr'] = df['tr'].rolling(window=window).mean()
    df.drop(columns=['tr'], inplace=True)
    return df

def calculate_fibonacci_retracement(df, low, high):
    """
    Calculate Fibonacci retracement levels.

    Args:
    df (pandas.DataFrame): DataFrame with 'low' and 'high' prices.
    low (float): Lowest point for the range.
    high (float): Highest point for the range.

    Returns:
    pandas.DataFrame: Original DataFrame with 'fib_0%', 'fib_23.6%', 'fib_38.2%', 'fib_61.8%', and 'fib_100%' columns added.
    """
    fib_levels = [0, 0.236, 0.382, 0.618, 1.0]
    for level in fib_levels:
        df[f'fib_{int(level * 100)}%'] = low + (level * (high - low))
    return df

def calculate_ichimoku_cloud(df):
    """
    Calculate Ichimoku Cloud components: Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B.

    Args:
    df (pandas.DataFrame): DataFrame with 'high' and 'low' prices.

    Returns:
    pandas.DataFrame: Original DataFrame with 'tenkan_sen', 'kijun_sen', 'senkou_span_a', and 'senkou_span_b' columns added.
    """
    tenkan_sen_period = 9
    kijun_sen_period = 26
    senkou_span_b_period = 52

    df['tenkan_sen'] = (df['high'].rolling(window=tenkan_sen_period).max() + df['low'].rolling(window=tenkan_sen_period).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(window=kijun_sen_period).max() + df['low'].rolling(window=kijun_sen_period).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_sen_period)
    df['senkou_span_b'] = ((df['high'].rolling(window=senkou_span_b_period).max() + df['low'].rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_sen_period)
    return df

def calculate_volume_profile(df, window=50):
    """
    Calculate Volume Profile.

    Args:
    df (pandas.DataFrame): DataFrame with 'volume' and 'close' prices.
    window (int): Window for calculating the volume profile (default is 50).

    Returns:
    pandas.DataFrame: Original DataFrame with 'volume_profile' column added.
    """
    df['volume_profile'] = df['volume'].rolling(window=window).sum()
    return df

def calculate_williams_percent_r(df, window=14):
    """
    Calculate Williams %R.

    Args:
    df (pandas.DataFrame): DataFrame with 'high' and 'low' prices.
    window (int): Window for calculating Williams %R (default is 14).

    Returns:
    pandas.DataFrame: Original DataFrame with 'williams_percent_r' column added.
    """
    highest_high = df['high'].rolling(window=window).max()
    lowest_low = df['low'].rolling(window=window).min()
    df['williams_percent_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    return df

def calculate_adx(df, window=14):
    """
    Calculate Average Directional Index (ADX).

    Args:
    df (pandas.DataFrame): DataFrame with 'high', 'low', and 'close' prices.
    window (int): Window for calculating ADX (default is 14).

    Returns:
    pandas.DataFrame: Original DataFrame with 'adx' column added.
    """
    df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), np.maximum(df['low'].shift(1) - df['low'], 0), 0)

    atr = df['tr'].rolling(window=window).mean()
    plus_dm = df['plus_dm'].rolling(window=window).mean()
    minus_dm = df['minus_dm'].rolling(window=window).mean()
    
    plus_di = (plus_dm / atr) * 100
    minus_di = (minus_dm / atr) * 100

    df['dx'] = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = df['dx'].rolling(window=window).mean()

    df.drop(columns=['tr', 'plus_dm', 'minus_dm', 'dx'], inplace=True)
    return df

def calculate_obv(df):
    """
    Calculate On-Balance Volume (OBV).

    Args:
    df (pandas.DataFrame): DataFrame with 'close' and 'volume' prices.

    Returns:
    pandas.DataFrame: Original DataFrame with 'obv' column added.
    """
    obv = [0]  # Initialize the first OBV value as 0
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i - 1]:
            obv.append(obv[-1] + df['volume'][i])
        elif df['close'][i] < df['close'][i - 1]:
            obv.append(obv[-1] - df['volume'][i])
        else:
            obv.append(obv[-1])  # If the closing price remains the same, OBV doesn't change

    df['obv'] = obv  # Assign the calculated OBV to the DataFrame
    return df

def calculate_chaikin_oscillator(df, ema_fast_period=3, ema_slow_period=10):
    """
    Calculate Chaikin Oscillator.

    Args:
    df (pandas.DataFrame): DataFrame with 'close' and 'volume' prices.
    ema_fast_period (int): Period for the fast EMA (default is 3).
    ema_slow_period (int): Period for the slow EMA (default is 10).

    Returns:
    pandas.DataFrame: Original DataFrame with 'chaikin_oscillator' column added.
    """
    ema_fast = df['close'].ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=ema_slow_period, adjust=False).mean()
    df['chaikin_oscillator'] = ema_fast - ema_slow
    return df

def calculate_pivot_points(df, pivot_type="standard"):
    """
    Calculate Pivot Points.

    Args:
    df (pandas.DataFrame): DataFrame with 'high', 'low', and 'close' prices.
    pivot_type (str): Type of pivot points (default is "standard", other types include "woodie" and "camarilla").

    Returns:
    pandas.DataFrame: Original DataFrame with pivot point levels added.
    """
    if pivot_type == "standard":
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df['support1'] = (2 * df['pivot_point']) - df['high']
        df['support2'] = df['pivot_point'] - (df['high'] - df['low'])
        df['resistance1'] = (2 * df['pivot_point']) - df['low']
        df['resistance2'] = df['pivot_point'] + (df['high'] - df['low'])
    elif pivot_type == "woodie":
        df['pivot_point'] = (df['high'] + df['low'] + (2 * df['close'])) / 4
        df['support1'] = (2 * df['pivot_point']) - df['high']
        df['support2'] = df['pivot_point'] - (df['high'] - df['low'])
        df['resistance1'] = (2 * df['pivot_point']) - df['low']
        df['resistance2'] = df['pivot_point'] + (df['high'] - df['low'])
    elif pivot_type == "camarilla":
        df['pivot_point'] = df['close']
        df['support1'] = df['close'] - (df['high'] - df['low']) * 1.1 / 12
        df['support2'] = df['close'] - (df['high'] - df['low']) * 1.1 / 6
        df['resistance1'] = df['close'] + (df['high'] - df['low']) * 1.1 / 12
        df['resistance2'] = df['close'] + (df['high'] - df['low']) * 1.1 / 6

    return df

def calculate_price_channels(df, window=20):
    """
    Calculate Price Channels (Donchian Channels).

    Args:
    df (pandas.DataFrame): DataFrame with 'high' and 'low' prices.
    window (int): Window for calculating the channels (default is 20).

    Returns:
    pandas.DataFrame: Original DataFrame with 'upper_channel' and 'lower_channel' columns added.
    """
    df['upper_channel'] = df['high'].rolling(window=window).max()
    df['lower_channel'] = df['low'].rolling(window=window).min()
    return df

def calculate_mass_index(df, window=9, ema_period=9):
    """
    Calculate Mass Index.

    Args:
    df (pandas.DataFrame): DataFrame with 'high' and 'low' prices.
    window (int): Window for calculating the Mass Index (default is 9).
    ema_period (int): Period for the EMA of the Mass Index (default is 9).

    Returns:
    pandas.DataFrame: Original DataFrame with 'mass_index' and 'ema_mass_index' columns added.
    """
    df['range'] = df['high'] - df['low']
    df['exponential_range'] = df['range'].ewm(span=window, adjust=False).mean()
    df['mass_index'] = df['exponential_range'] / df['exponential_range'].shift(1)
    df['ema_mass_index'] = df['mass_index'].ewm(span=ema_period, adjust=False).mean()
    df.drop(columns=['range', 'exponential_range'], inplace=True)
    return df

def calculate_elliott_wave(df):
    """
    Calculate Elliott Wave patterns based on price data.

    Args:
    df (pandas.DataFrame): DataFrame with 'close' prices.

    Returns:
    pandas.DataFrame: Original DataFrame with 'elliott_wave' column indicating the Elliott Wave pattern.
    """
    # Sample implementation (simplified)
    df['elliott_wave'] = 0  # Initialize the column

    for i in range(4, len(df)):
        # Define criteria for wave patterns (simplified for illustration)
        if (df['close'][i] > df['close'][i - 2]) and (df['close'][i - 1] > df['close'][i - 3]):
            df.at[i, 'elliott_wave'] = 1  # Indicates an upward wave
        elif (df['close'][i] < df['close'][i - 2]) and (df['close'][i - 1] < df['close'][i - 3]):
            df.at[i, 'elliott_wave'] = -1  # Indicates a downward wave

    return df
            
def calculate_correlation(pair1_data, pair2_data):
    """
    Calculate the Pearson correlation coefficient between two cryptocurrency pairs.

    Parameters:
    - pair1_data: A pandas Series or DataFrame with historical price data for the first cryptocurrency pair.
    - pair2_data: A pandas Series or DataFrame with historical price data for the second cryptocurrency pair.

    Returns:
    - correlation: The Pearson correlation coefficient between the two pairs.
    """
    # Convert the dictionary to a pandas DataFrame
    df_1 = pd.DataFrame(list(pair1_data.items()), columns=["index", "close"])
    df_2 = pd.DataFrame(list(pair2_data.items()), columns=["index", "close"])
    # Ensure that both dataframes have the same index (date) and non-null values
    df_1.dropna(inplace=True)
    df_2.dropna(inplace=True)
    
    # Merge the two dataframes based on the date index
    merged_data = pd.concat([df_1, df_2], axis=1, join='inner')
    
    # Calculate the Pearson correlation coefficient
    correlation = merged_data.corr().iloc[0, 1]

    return correlation

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

# Function to process pair data
def process_pair(exchange, pair, candle_type, history_limit, indicator_functions, indicators):
    try:
        bars = exchange.fetch_ohlcv(pair, timeframe=candle_type, limit=history_limit)

        df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Apply each indicator function to the DataFrame
        for indicator_function in indicator_functions:
            df = indicator_function(df)

        bollinger_thresholds = calculate_bollinger_bands_thresholds(df, periods=5)
        ichimoku_thresholds = calculate_ichimoku_cloud_thresholds(df)
        macd_thresholds = calculate_macd_thresholds(df)
        obv_thresholds = calculate_obv_thresholds(df)

        # Define threshold values for each indicator
        thresholds = {
            'rsi': (30, 70),  # RSI threshold values
            'macd': macd_thresholds,  # MACD threshold values
            'chaikin_oscillator': (-0.2, 0.2),  # Chaikin Oscillator threshold values
            'bollinger_bands': bollinger_thresholds,  # Bollinger Bands threshold values
            'atr': (14, 35),  # ATR threshold values
            'stoch': (20, 80),  # Stochastic Oscillator threshold values
            'ichimoku_cloud': ichimoku_thresholds,  # Ichimoku Cloud threshold values
            'williams_percent_r': (20, 80),  # Williams %R threshold values
            'adx': (25, 50),  # ADX threshold values
            'obv': obv_thresholds,  # On-Balance Volume threshold values
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

        data_by_candle_type = df.to_dict()

        return {
            "pair": pair,
            "candle_type": candle_type,
            "data": data_by_candle_type
        }

    except Exception as e:
        print(f"An error occurred for {pair}, {candle_type}: {e}")
        return None

def is_position_risky(df, var_threshold, candles_to_consider):

    # Calculate the rolling minimum of the 'low' column over the specified number of candles
    df['rolling_min_low'] = df['low'].rolling(window=candles_to_consider, min_periods=1).min()
    df['rolling_max_high'] = df['high'].rolling(window=candles_to_consider, min_periods=1).max()

    # Calculate the variation from the rolling minimum to the 'close' price
    df['price_variation'] = (df['close'] - df['rolling_min_low']) / df['rolling_min_low']

    # Check if any row has a variation exceeding the specified threshold (expressed as a percentage)
    is_long_risky = any(df['price_variation'] > var_threshold / 100)
    is_short_risky = any(df['price_variation'] < -var_threshold / 100)

    return {"LONG": is_long_risky, "SHORT": is_short_risky}
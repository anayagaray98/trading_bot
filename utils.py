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

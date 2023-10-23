import ccxt
import config
import pandas as pd
from datetime import datetime
import warnings
from .utils import get_ema, get_macd, stochrsi, calculate_adx, \
    calculate_obv, calculate_chaikin_oscillator, calculate_pivot_points, \
        calculate_price_channels, calculate_mass_index, calculate_elliott_wave
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)


""" Configuration Variables """

pairs = ['DOTUP/USDT', 'DOGE/USDT', 'SHIB/USDT']
exchange = ccxt.binance({
    "apiKey": config.API_KEY,
    "secret": config.API_SECRET
})


def place_order(symbol, quantity, side, price, order_type=ccxt.binance.ORDER_TYPE_LIMIT):
    try:
        print(f"Placing {side} order for {quantity} {symbol} at price {price}")
        order = exchange.create_limit_order(symbol, order_type, side, quantity, price)
        print("Order details:")
        print(order)
    except Exception as e:
        print(f"An error occurred while placing the order: {e}")


def get_model_signal(model, X):
    # Use the model to predict the trading signal
    signal = model.predict(X)
    return signal


def run_bot():
    for pair in pairs:
        print(f"Fetching new bars for {datetime.now().isoformat()}")
        bars = exchange.fetch_ohlcv(pair, timeframe='15m', limit=100)
        df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate technical indicators
        df = stochrsi(df)
        df = get_macd(df)
        df = get_ema(df)
        df = calculate_adx(df)  # Average Directional Index
        df = calculate_obv(df)  # On-Balance Volume
        df = calculate_chaikin_oscillator(df)  # Chaikin Oscillator
        df = calculate_pivot_points(df)  # Pivot Points
        df = calculate_price_channels(df)  # Donchian Channels
        df = calculate_mass_index(df)  # Mass Index
        df = calculate_elliott_wave(df) 

        # Generate trading signals based on the indicators
        # Example: Buy when RSI crosses above 30, MACD crosses above its signal line, and Chaikin Oscillator is positive.
        df['buy_signal'] = (
            (df['rsi'] > 30) &
            (df['rsi'].shift(1) <= 30) &
            (df['macd'] > df['signal']) &
            (df['macd'].shift(1) <= df['signal'].shift(1)) &
            (df['chaikin_oscillator'] > 0)
        )

        # Example: Sell when RSI crosses below 70, MACD crosses below its signal line, and Mass Index is above a certain threshold.
        df['sell_signal'] = (
            (df['rsi'] < 70) |
            (df['rsi'].shift(1) >= 70) |
            (df['macd'] < df['signal']) |
            (df['macd'].shift(1) >= df['signal'].shift(1)) |
            (df['ema'] < df['close'])
        )

        # Define the target variable: 1 for Buy, -1 for Sell, 0 for Hold
        df['action'] = 0
        df.loc[df['buy_signal'], 'action'] = 1
        df.loc[df['sell_signal'], 'action'] = -1

        # Prepare the feature matrix and target vector
        features = ['rsi', 'macd', 'signal', 'stoch_k', 'stoch_d']
        df[features] = df[features].fillna(0)  # Handle missing values
        X = df[features]
        y = df['action']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train an ML model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = model.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Get the trading signal from the model
        signal = get_model_signal(model, X)

        # TRADING PART RIGHT HERE
        # Example: Place a limit order to buy 1 unit of the asset if the model predicts a buy signal

        # Calculate the target price based on the model's predictions
        current_price = df['close'].iloc[-1]
        if signal == 1:
            # If the model predicts a buy signal, you may consider a target price higher than the current price.
            target_price = current_price * 1.05  # You can adjust the multiplier as needed.
            place_order(pair, 1, 'buy', target_price)
        elif signal == -1:
            # If the model predicts a sell signal, you may consider a target price lower than the current price.
            target_price = current_price * 0.95  # You can adjust the multiplier as needed.
            place_order(pair, 1, 'sell', target_price)
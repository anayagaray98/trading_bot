import ccxt
import config
import pandas as pd
from datetime import datetime
import warnings
from utils import calculate_ema, calculate_macd, calculate_stochrsi, calculate_adx, \
    calculate_obv, calculate_chaikin_oscillator, calculate_pivot_points, \
        calculate_price_channels, calculate_mass_index, calculate_elliott_wave
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np

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
exchange.verbose = True  # debug output


# List of machine learning models to evaluate
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42)
}

best_model = None
best_accuracy = 0

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

def get_model_signal(model, X):
    # Use the model to predict the trading signal
    signal = model.predict(X)
    return signal

def run_bot():
    for pair in pairs:
        print(f"Fetching new bars for {datetime.now().isoformat()}")
        bars = exchange.fetch_ohlcv(pair, timeframe=candle_type, limit=history_limit)
        df = pd.DataFrame(bars[:-1], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Calculate technical indicators
        df = calculate_stochrsi(df)
        df = calculate_macd(df)
        df = calculate_ema(df)
        df = calculate_adx(df)  # Average Directional Index
        df = calculate_chaikin_oscillator(df)  # Chaikin Oscillator
        df = calculate_pivot_points(df)  # Pivot Points
        df = calculate_price_channels(df)  # Donchian Channels
        df = calculate_mass_index(df)  # Mass Index
        df = calculate_elliott_wave(df) 

        # technical indicators to use
        indicators = ['rsi', 'macd', 'chaikin_oscillator', 'ema_mass_index', 'elliott_wave']

        # Define threshold values for each indicator
        thresholds = {
            'rsi': (30, 70),  # Buy when RSI is above 70 and sell when RSI is below 30
            'macd': (0, 0),   # Buy when MACD is above the signal line and sell when below
            'chaikin_oscillator': (0, 0),  # Buy when Chaikin Oscillator is above 0 and sell when below
            'ema_mass_index': (1.5, 0.8),  
            'elliott_wave': (0, 0)  
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
                df.at[i, 'sell_signal'] = -1
                df.at[i, 'sell_signal_confidence'] = bearish_indicators / len(indicators)
            else:
                df.at[i, 'sell_signal'] = 0
                df.at[i, 'sell_signal_confidence'] = 0.0  # No sell signal

        # Define the target variable: 1 for Buy, -1 for Sell, 0 for Hold
        # df['action'] = 0
        # df.loc[df['buy_signal'], 'action'] = 1
        # df.loc[df['sell_signal'], 'action'] = -1
        # df['action'] = df['action'].replace(0, np.nan)  # Replace 0 (Hold) with NaN
        # df['action'].fillna(method='ffill', inplace=True)  # Forward-fill NaN values to prioritize the most recent signal

        df.to_csv(f"data/trading_info_for_model_{pair.replace('/', '_')}.csv")

        # # Prepare the feature matrix and target vector
        # features = ['rsi', 'macd', 'signal', 'srsi_k', 'srsi_d']
        # df[features] = df[features].fillna(0)  # Handle missing values
        # X = df[features]
        # y = df['action']

        # # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # best_model = None
        # best_accuracy = 0
        # for model_name, model in models.items():
        #     # Create and train an ML model
        #     model.fit(X_train, y_train)

        #     # Make predictions on the testing set
        #     y_pred = model.predict(X_test)

        #     # Calculate the accuracy of the model
        #     accuracy = accuracy_score(y_test, y_pred)
        #     print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

        #     # Check if this model is the best so far
        #     if accuracy > best_accuracy:
        #         best_model = model
        #         best_accuracy = accuracy
        
        # print("*"*50)
        # print("Best Model:", best_model)

        # Train the best model on the entire dataset
        # best_model.fit(X, y)

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
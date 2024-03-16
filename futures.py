import os
import ccxt
import config
import pandas as pd
import numpy as np
import warnings
import schedule
import json
from dateutil.relativedelta import relativedelta
from datetime import datetime
import concurrent.futures
from utils import *

warnings.filterwarnings('ignore')


""" Configuration Variables """

pairs = ['ALPHA/USDT', 'ADA/USDT', 'MATIC/USDT', 'FLM/USDT', 'REEF/USDT', 'XRP/USDT', 'GALA/USDT', 'WAXP/USDT', 
         'C98/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'UNI/USDT','ATOM/USDT', 'ARPA/USDT', 'ONT/USDT', 'STMX/USDT', 
         'RLC/USDT', 'SXP/USDT','ICX/USDT', 'FIL/USDT'] 

candle_types = ['1m', '5m'] # For intraday strategy.
history_limit = 100 # 1500 is the largest size per API call.
allowed_confidence_threshold = 0.65 # This is the minimum confidence level to make a buy/sell decision.
trade_quantity_amount = 80.00 # Quantity in USDT.
leverage = 3 # Leverage multiplier.
type_of_order = 'market' # limit, market.
expected_return_level = 5 # %
stop_loss_level = 1.5 # %
max_allowed_positions = 2 # Number of positions allowed in the trading strategy.
max_correlation_value = 80 # %
recently_traded_cryptos_path = "trades_history/futures_traded_cryptos.json"
hours_number_until_trade_again = 5 # Number of hours to wait until asset can be tradable again.
batch_size = 10  # Number of pairs to process in each batch.
var_threshold = 5 # %. Max variation allowed before placing a trade.
candles_to_consider = 50 # Number of candles to consider when calculating the var_threshold
num_cores = os.cpu_count() # Get the number of CPU cores.

exchange = ccxt.binance({
    "apiKey": config.API_KEY_PRODUCTION,
    "secret": config.API_SECRET_PRODUCTION,
    'options': {
        'defaultType': 'future',
    },
})

exchange.verbose = False  # debug output

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

def get_open_futures_positions():
    try:
        balance = exchange.fetch_balance()
        positions = balance['info']['positions']
        open_positions = [position for position in positions if float(position['positionAmt']) != 0]
        return open_positions
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

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

def run_bot():

    print("\nProcessing data...\n")
    print(f"\nNumber of cores: {num_cores}\n")

    results = []
    trades_structure = []
    # Create a pool of worker processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores*4 or 4) as executor:
        for candle_type in candle_types:
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                futures = {executor.submit(process_pair, exchange, pair, candle_type, history_limit, indicator_functions, indicators): pair for pair in batch_pairs}
                for future in concurrent.futures.as_completed(futures):
                    pair = futures[future]
                    result = future.result()
                    results.append(result)
    
    # Process the results to construct trades_structure
    for result in results:
        if result:
            if not any(x['pair'] == result['pair'] for x in trades_structure):
                trade_structure_obj = {}
                trade_structure_obj['pair'] = result['pair']
                trade_structure_obj['candle_types'] = {result['candle_type']: result['data']}
                trades_structure.append(trade_structure_obj)
            else:
                for x in range(len(trades_structure)):
                    if trades_structure[x]['pair'] == result['pair']:
                        trades_structure[x]['candle_types'][result['candle_type']] = result['data']


    """ Calculate Correlations """

    print("\nHold on! Calculating correlations...\n")

    correlations = []
    for candle_type in candle_types: 
        for j in range(len(trades_structure)):
            pair1 = trades_structure[j]['pair']
            pair1_data = trades_structure[j]['candle_types'][candle_type]['close']
            for k in range(len(trades_structure)):
                pair2 = trades_structure[k]['pair']
                pair2_data = trades_structure[k]['candle_types'][candle_type]['close']
                correlation_value = calculate_correlation(pair1_data, pair2_data)
                correlations.append({"pair_1": pair1, "pair_2": pair2, "corr": correlation_value, "candle_type": candle_type})
    
    # Create a dictionary to accumulate correlation values for each unique pair
    pair_correlations = {}

    # Iterate over the correlations
    for correlation in correlations:
        pair_1 = correlation["pair_1"]
        pair_2 = correlation["pair_2"]
        corr = correlation["corr"]

        # Check if this pair combination is already in the dictionary
        if (pair_1, pair_2) in pair_correlations:
            pair_correlations[(pair_1, pair_2)].append(corr)
        elif (pair_2, pair_1) in pair_correlations:
            pair_correlations[(pair_2, pair_1)].append(corr)
        else:
            pair_correlations[(pair_1, pair_2)] = [corr]

    # Calculate the mean correlation for each unique pair
    pair_means = {}
    for pair, corr_values in pair_correlations.items():
        mean_corr = np.mean(corr_values)
        pair_means[pair] = mean_corr

    for i in range(len(trades_structure)):

        print()
        print("*"*20, ' ' + trades_structure[i]['pair'] + ' ', "*"*20)
        print()
        
        signals = []
        
        # Get current price
        last_price_bars = exchange.fetch_ticker(trades_structure[i]['pair'])
        last_price = last_price_bars['last']
        current_price = float(last_price)
        risky_position_obj = {"LONG": False, "SHORT": False}

        for candle_type in candle_types: 

            df = pd.DataFrame(trades_structure[i]['candle_types'][candle_type])

            if candle_type == '5m':
                risky_position_obj = is_position_risky(df, var_threshold, candles_to_consider)
            
            # Get the last 5 rows of the DataFrame
            last_5_rows = df[['buy_signal', 'buy_signal_confidence', 'sell_signal', 'sell_signal_confidence']].tail(5)
            df = pd.DataFrame()

            # Extract the last row for 'buy_signal' and 'sell_signal'
            last_row = last_5_rows.iloc[-1]

            buy_signal = last_row['buy_signal']
            sell_signal = last_row['sell_signal']

            # Calculate the mean confidence for the last 5 rows
            buy_confidence = last_5_rows['buy_signal_confidence'].mean()
            sell_confidence = last_5_rows['sell_signal_confidence'].mean()

            if buy_signal == 1:
                if buy_confidence > allowed_confidence_threshold:
                    signals.append('buy')
                else:
                    signals.append(None)

            elif buy_signal == 0:
                if sell_signal == 0:
                    signals.append(None)
                elif sell_signal == 1:
                    if sell_confidence > allowed_confidence_threshold:
                        signals.append('sell')
                    else:
                        signals.append(None)
        
        set_leverage(leverage, trades_structure[i]['pair'])
        pair_positions = get_account_positions(trades_structure[i]['pair'])

        in_position = False
        set_position_side = None
        set_position_amount = 0.0
        has_notional = False
        set_entry_price = 0.0

        for position in pair_positions:
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
    
        # Being aware of current price change to trigger stop loss and take profit
        if in_position:
            price_change = ((current_price/set_entry_price)-1)*100
            print(f"CURRENT PRICE CHANGE: {round(price_change, 2)} %")
        
        if set_position_side:
            print(f"Open Order: {set_position_side}")
        else:
            print("There is no opened position.") 

        all_open_positions = get_open_futures_positions()

        # Triggering stop loss or take profit
        if in_position and not has_notional:                
            if set_position_side == 'SHORT':
                if (price_change <= (expected_return_level * -1)):
                    print("Triggering TAKE PROFIT")
                if (price_change >= stop_loss_level):
                    print("Triggering STOP LOSS")

                if (price_change <= (expected_return_level * -1)) or (price_change >= stop_loss_level):

                    params = {'positionSide': 'SHORT', 'leverage':leverage}

                    with open(recently_traded_cryptos_path, 'r') as json_file:
                        data = json.load(json_file)
                    
                    try:
                        place_order(symbol=trades_structure[i]['pair'], quantity=set_position_amount, side='sell', price=current_price, order_type=type_of_order, params=params)
                        
                        pair_found = False
                        for pair in data['pairs']:
                            if trades_structure[i]['pair'] == pair['symbol']:
                                pair['closed_time'] = datetime.now().timestamp()
                                pair_found = True
                                break

                        if not pair_found:
                            data['pairs'].append({"symbol":trades_structure[i]['pair'], "closed_time":datetime.now().timestamp()})
                        
                        with open(recently_traded_cryptos_path, 'w') as json_file:
                            json.dump(data, json_file, indent=4)

                    except Exception as e:
                        print(f"An error occured: {e}")
            
            if set_position_side == 'LONG':
                if (price_change >= expected_return_level):
                    print("Triggering TAKE PROFIT")
                if (price_change <= (stop_loss_level * -1)):
                    print("Triggering STOP LOSS")

                if (price_change >= expected_return_level) or (price_change <= (stop_loss_level * -1)):

                    params = {'positionSide': 'LONG', 'leverage':leverage}

                    with open(recently_traded_cryptos_path, 'r') as json_file:
                        data = json.load(json_file)
                    
                    try:
                        place_order(symbol=trades_structure[i]['pair'], quantity=set_position_amount, side='sell', price=current_price, order_type=type_of_order, params=params)
                        
                        pair_found = False
                        for pair in data['pairs']:
                            if trades_structure[i]['pair'] == pair['symbol']:
                                pair['closed_time'] = datetime.now().timestamp()
                                pair_found = True
                                break

                        if not pair_found:
                            data['pairs'].append({"symbol":trades_structure[i]['pair'], "closed_time":datetime.now().timestamp()})

                        with open(recently_traded_cryptos_path, 'w') as json_file:
                            json.dump(data, json_file, indent=4)

                    except Exception as e:
                        print(f"An error occured: {e}")

        if final_signal:    
            print(f"{str(final_signal).upper()} "*5)
            if in_position and not has_notional:  
                # If stop loss and take profit are not triggered, check for contrary signals and close if it's the case
                if final_signal == 'buy' and set_position_side == 'SHORT':
                    print(f"Trying to cancel an offer: {set_position_side}")
                    params = {'positionSide': 'SHORT', 'leverage':leverage}
                    target_price = current_price * 0.99
                    place_order(symbol=trades_structure[i]['pair'], quantity=set_position_amount, side='sell', price=target_price, order_type=type_of_order, params=params)
                
                if final_signal == 'sell' and set_position_side == 'LONG':
                    print(f"Trying to cancel an offer: {set_position_side}")
                    params = {'positionSide': 'LONG', 'leverage':leverage}
                    target_price = current_price * 1.01
                    place_order(symbol=trades_structure[i]['pair'], quantity=set_position_amount, side='sell', price=target_price, order_type=type_of_order, params=params)

            break_trading_process = False
            # TRADING PART RIGHT HERE
            if not in_position and not has_notional:

                with open(recently_traded_cryptos_path, 'r') as json_file:
                    data = json.load(json_file)

                if len(data['pairs']) > 0:
                    for traded_pair in data['pairs']:
                        if traded_pair['symbol'] == trades_structure[i]['pair']:
                            if traded_pair['closed_time']:
                                closed_time = datetime.fromtimestamp(traded_pair['closed_time'])
                                desired_timedelta = relativedelta(hours=hours_number_until_trade_again)
                                print(f"Closed time: {closed_time}", f"Due time: {closed_time + desired_timedelta}")
                                if datetime.now() <= closed_time + desired_timedelta:
                                    print(f"This pair cannot be traded, wait until {closed_time + desired_timedelta}, so that you can trade it")
                                    break_trading_process = True

                if not break_trading_process:
                    if len(all_open_positions) < max_allowed_positions:
                        
                        corr_value = None  # Initialize corr_value to None
                        if len(all_open_positions) > 0:
                            for open_position in all_open_positions:
                                open_position_symbol = open_position['symbol'].replace("USDT", "/USDT")
                                key1 = (f"{trades_structure[i]['pair']}", f"{open_position_symbol}")
                                key2 = (f"{open_position_symbol}", f"{trades_structure[i]['pair']}")

                                if key1 in pair_means.keys():
                                    corr_value = pair_means[key1]
                                    break

                                elif key2 in pair_means.keys():
                                    corr_value = pair_means[key2]
                                    break
                        
                        if not risky_position_obj['LONG' if final_signal == 'buy' else 'SHORT']:
                            if not len(all_open_positions) == 0:

                                if (corr_value):
                                    if (abs(corr_value) <= max_correlation_value/100):
                                        print(f"Trying to create an offer: {final_signal}")
                                        if final_signal == 'buy':
                                            params = {'positionSide': 'LONG', 'leverage':leverage}
                                            new_quantity = (trade_quantity_amount * leverage) / current_price
                                            target_price = current_price * 0.99
                                            place_order(symbol=trades_structure[i]['pair'], quantity=new_quantity, side='buy', price=target_price, order_type=type_of_order, params=params)
                                        else:
                                            params = {'positionSide': 'SHORT', 'leverage':leverage}
                                            new_quantity = (trade_quantity_amount * leverage) / current_price
                                            target_price = current_price * 1.01
                                            place_order(symbol=trades_structure[i]['pair'], quantity=new_quantity, side='sell', price=target_price, order_type=type_of_order, params=params)
                                    else:
                                        print(f"This pair: {trades_structure[i]['pair']} is highly correlated with an open position. We won't move forward.")
                            
                            else:
                                print(f"Trying to create an offer: {final_signal}")
                                if final_signal == 'buy':
                                    params = {'positionSide': 'LONG', 'leverage':leverage}
                                    new_quantity = (trade_quantity_amount * leverage) / current_price
                                    target_price = current_price * 0.99
                                    place_order(symbol=trades_structure[i]['pair'], quantity=new_quantity, side='buy', price=target_price, order_type=type_of_order, params=params)
                                else:
                                    params = {'positionSide': 'SHORT', 'leverage':leverage}
                                    new_quantity = (trade_quantity_amount * leverage) / current_price
                                    target_price = current_price * 1.01
                                    place_order(symbol=trades_structure[i]['pair'], quantity=new_quantity, side='sell', price=target_price, order_type=type_of_order, params=params)
                        else:
                            print("This is a risky position. We are leaving it aside.")

                    else:
                        print("You cannot add another pair to your trades. Wait until they close.")
        else:
            print("Nothing to do. There is no significant signal here.")

        print()
        print("_"*120)
        print()

if __name__ == "__main__":
    schedule.every(5).seconds.do(run_bot)

    while True:
        schedule.run_pending()

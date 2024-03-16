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

pairs = ['FTT/USDT'] 

candle_types = ['5m', '1h'] # For intraday strategy.
history_limit = 100 # 1500 is the largest size per API call.
margin_mode = 'cross' # it can be cross or isolated
allowed_confidence_threshold = 0.65 # This is the minimum confidence level to make a buy/sell decision.
trade_quantity_amount = 500.00 # Quantity in USDT.
leverage = 3 # Leverage multiplier.s
type_of_order = 'market' # limit, market.
expected_return_level = 5 # %
stop_loss_level = 1.5 # %
max_allowed_positions = 2 # Number of positions allowed in the trading strategy.
max_correlation_value = 80 # %
recently_traded_cryptos_path = "spot_margin_traded_cryptos.json"
hours_number_until_trade_again = 5 # Number of hours to wait until asset can be tradable again.
batch_size = 10  # Number of pairs to process in each batch.
var_threshold = 5 # %. Max variation allowed before placing a trade.
candles_to_consider = 50 # Number of candles to consider when calculating the var_threshold
num_cores = os.cpu_count() # Get the number of CPU cores.

exchange = ccxt.binance({
    "apiKey": config.API_KEY_PRODUCTION,
    "secret": config.API_SECRET_PRODUCTION,
})

exchange.verbose = False  # debug output

def place_order(symbol, quantity, side, price, order_type, extra_params={}):
    try:
        print(f"Placing {side} order for {quantity} {symbol} at price {price}")

        params = {'margin': True, 'marginMode': margin_mode}

        for key, value in extra_params:
            params[key] = value

        if order_type == 'limit':
            order = exchange.create_order(symbol=symbol, type=order_type, side=side, amount=quantity, price=price, params=params)

        elif order_type == 'market':
            order = exchange.create_order(symbol=symbol, type=order_type, side=side, amount=quantity, params=params)

        # If operation type is buy then save to traded assets or update if it is a sell operation with closed time
        with open(recently_traded_cryptos_path, 'r') as json_file:
            data = json.load(json_file)
            
        try:                
            pair_found = False
            for pair in data['pairs']:
                if symbol == pair['symbol']:
                    pair['closed_time'] = datetime.now().timestamp() if side == 'sell' else None
                    pair['entry_price'] = price
                    pair_found = True
                    break

            if not pair_found:
                data['pairs'].append({"symbol":symbol, "closed_time":datetime.now().timestamp() if side == 'sell' else None, "entry_price":price})

            with open(recently_traded_cryptos_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

        except Exception as e:
            print(f"An error occured: {e}")

        print("Order details:")
        print(order)

    except Exception as e:
        print(f"An error occurred while placing the order: {e}")

def get_position(pair):

    balance = exchange.fetch_balance({'type': 'spot', 'marginMode':margin_mode})
    positions = balance['info']['userAssets']
    asset = pair.split('/')[0]

    for position in positions:
        if position['asset'] == asset:
            return position

def get_account_positions():

    balance = exchange.fetch_balance({'type': 'spot', 'marginMode':margin_mode})
    positions = balance['info']['userAssets']
    open_positions = [position for position in positions if float(position['netAsset']) > 0]

    return open_positions

def margin_borrow(pair):

    try:
        asset_code = pair.split("/")[0]
        currency = exchange.currency(asset_code)
        exchange.sapi_post_margin_loan(
            {
                'asset': currency['id'],
                'amount': exchange.currency_to_precision(asset_code, trade_quantity_amount*leverage)
            }
        )
    except ccxt.InsufficientFunds as e:
        print('Margin loan failed: not enough funds')
        print(str(e))

    except Exception as e:
        print('Margin loan failed')
        print(str(e))

def margin_repay(pair):
    try:
        asset_code = pair.split("/")[0]
        currency = exchange.currency(asset_code)
        exchange.sapi_post_margin_repay(
            {
                'asset': currency['id'],
                'amount': exchange.currency_to_precision(asset_code, trade_quantity_amount*leverage)
            }
        )

    except Exception as e:
        print('Margin repay failed')
        print(str(e))

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
        

        pair_position = get_position(trades_structure[i]['pair'])

        in_position = False
        position_amt = float(pair_position['netAsset'])

        if position_amt > 0:
            in_position = True

        final_signal = 'buy' if all(signal == 'buy' for signal in signals) else 'sell' if all(signal == 'sell' for signal in signals) else None
    
        # Being aware of current price change to trigger stop loss and take profit
        if in_position:
                
            if final_signal == 'sell':
                place_order(symbol=trades_structure[i]['pair'], quantity=trade_quantity_amount, side='sell', price=current_price, order_type=type_of_order)

            else:

                with open(recently_traded_cryptos_path, 'r') as json_file:
                    data = json.load(json_file)
            
                if not (len([traded_asset for traded_asset in data['pairs'] if traded_asset['symbol'] == trades_structure[i]['pair']]) > 0) or not ([traded_asset for traded_asset in data['pairs'] if traded_asset['symbol'] == trades_structure[i]['pair']][0]['entry_price']):
                    print(f"The {trades_structure[i]['pair']} does not exist in the traded assets file.")
                    continue

                entry_price = [traded_asset for traded_asset in data['pairs'] if traded_asset['symbol'] == trades_structure[i]['pair']][0]['entry_price']

                price_change = ((current_price/entry_price)-1)*100
                print(f"Current price change: {round(price_change, 2)} %")

                # Triggering stop loss or take profit
                if (price_change >= expected_return_level):
                    print("Triggering TAKE PROFIT")

                if (price_change <= (stop_loss_level * -1)):
                    print("Triggering STOP LOSS")

                if (price_change >= expected_return_level) or (price_change <= (stop_loss_level * -1)):
                        
                    try:
                        place_order(symbol=trades_structure[i]['pair'], quantity=position_amt, side='sell', price=current_price, order_type=type_of_order)

                    except Exception as e:
                        print(f"An error occured: {e}")

        else:
            if final_signal == 'sell':
                print("No open position. Nothing to sell.") 
                continue
            
            elif final_signal == 'buy':
                # Placing buy offer
                print(f"Trying to buy: {trades_structure[i]['pair']}")

                # Verifying it is not a recent traded crypto
                with open(recently_traded_cryptos_path, 'r') as json_file:
                    data = json.load(json_file)
                
                break_trading_process = False
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

                    all_open_positions = get_account_positions()

                    if len(all_open_positions) < max_allowed_positions:

                        corr_value = None  # Initialize corr_value to None
                        if len(all_open_positions) > 0:
                            for open_position in all_open_positions:
                                open_position_symbol = f"{open_position['asset']}/USDT"
                                key1 = (f"{trades_structure[i]['pair']}", f"{open_position_symbol}")
                                key2 = (f"{open_position_symbol}", f"{trades_structure[i]['pair']}")

                                if key1 in pair_means.keys():
                                    corr_value = pair_means[key1]
                                    break

                                elif key2 in pair_means.keys():
                                    corr_value = pair_means[key2]
                                    break
                        
                        if not risky_position_obj['LONG']:

                            if corr_value:
                                if (abs(corr_value) <= max_correlation_value/100):
                                    place_order(symbol=trades_structure[i]['pair'], quantity=trade_quantity_amount, side='buy', price=current_price, order_type=type_of_order)

                                else:
                                    print(f"This pair: {trades_structure[i]['pair']} is highly correlated with an open position. We won't move forward.")
                                    continue
                            else:
                                place_order(symbol=trades_structure[i]['pair'], quantity=trade_quantity_amount, side='buy', price=current_price, order_type=type_of_order)
                        else:
                            print("This is a risky position. We are leaving it aside.")
                            continue

                    else:
                        print("You cannot add another pair to your trades. Wait until one of those closes.")
                        continue
            else:
                continue

        print()
        print("_"*120)
        print()

if __name__ == "__main__":
    schedule.every(5).seconds.do(run_bot)

    while True:
        schedule.run_pending()

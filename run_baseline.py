# preprocess table and run baseline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from eth_env import ETHTradingEnv
from argparse import Namespace

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


device = 'cuda:7'
BUY, SELL = 0.5, -0.5
# BUY, SELL = 1, -1
FULL_BUY, FULL_SELL = 1, -1
strategies = ['SMA', 'MACD']
# strategies = ['SMA', 'MACD', 'SLMA', 'BollingerBands', 'buy_and_hold', 'optimal', 'LSTM', 'Multimodal']
sma_periods = [5, 10, 15, 20, 30]


# # eth bull dates
dates = ['2023-02-01','2024-01-24', '2024-03-13']

# sol bear dates
# dates = ['2023-02-01','2024-05-21', '2024-07-11']

# sol bull dates
# dates = ['2023-02-01','2024-01-24', '2024-03-13']

VAL_START, VAL_END = dates[-3], dates[-2]
TEST_START, TEST_END = dates[-2], dates[-1]
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


df = pd.read_csv('data/Bitcoin_31_10_2023-22_07_2024.csv')
df['date'] = pd.to_datetime(df['timestamp'])

# SMA
for period in sma_periods:
    df[f'SMA_{period}'] = df['open'].rolling(window=period).mean()
    df[f'STD_{period}'] = df['open'].rolling(window=period).std()

# MACD and Signal Line
df['EMA_12'] = df['open'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['open'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# dataset stats
for mi in range(len(dates)-1):
    starting_date = dates[mi]
    ending_date = dates[mi+1]
    y, m, _ = starting_date.split('-')
    df_m = df[(df['date'] >= starting_date) & (df['date'] <= ending_date)]
    print(f'{starting_date} to {ending_date} length:', len(df_m))
    stat = [df_m.iloc[0]['open'], df_m['open'].max(), df_m['open'].min(), df_m.iloc[-1]['open']]
    print('open, max, min, close:', [f'{s:.2f}' for s in stat])
    # df_m.to_csv(f'data/eth_f'{y}{m}'.csv', index=False)
print()

# # create dataset code for lstm
# def create_dataset(dataset, look_back=1):
#     X, Y = [], []
#     for i in range(len(dataset)-look_back):
#         a = dataset[i:(i+look_back), 0]
#         X.append(a)
#         Y.append(dataset[i + look_back, 0])
#     return np.array(X), np.array(Y)

def create_dataset(dataset, look_back=5):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32).view(-1, 1)

def run_strategy(strategy, sargs):
    env = ETHTradingEnv(Namespace(starting_date=sargs['starting_date'], ending_date=sargs['ending_date'], dataset='eth'))
    df_tmp = df[(df['date'] >= sargs['starting_date']) & (df['date'] <= sargs['ending_date'])]
    df_tmp.reset_index(drop=True, inplace=True)
    state, reward, done, info = env.reset()  # only use env to act and track profit

    starting_net_worth = state['net_worth']
    irrs = []
    previous_signal = None  # Track the previous day signal
    previous_net_worth = starting_net_worth
    # Iterate through each row in the DataFrame to simulate trading

    for index, row in df_tmp.iterrows():
        open_price = state['open']
        cash = state['cash']
        eth_held = state['eth_held']
        net_worth = state['net_worth']
        date = row['date']
        y, m, d = date.year, date.month, date.day
        irrs.append((net_worth / previous_net_worth) - 1)
        previous_net_worth = net_worth
        if done:
            break

        if strategy == 'SMA':
            period = sargs['period']
            sma_column = f'SMA_{period}'
            current_signal = 'hold'
            if open_price > row[sma_column]:  # golden cross?
                # current_signal = 'sell'
                current_signal = 'buy'
            elif open_price < row[sma_column]:  # death cross?
                # current_signal = 'buy'
                current_signal = 'sell'
                
            action = 0
            # if current_signal != previous_signal:
            if True:
                if current_signal == 'buy' and cash > 0:
                    action = BUY
                elif current_signal == 'sell' and eth_held > 0:
                    action = SELL
            previous_signal = current_signal
                
        elif strategy == 'SLMA':
            short = sargs['short']
            long = sargs['long']
            current_signal = 'hold'
            if row[short] > row[long]:  # golden cross?
                current_signal = 'buy'
            elif row[short] < row[long]:  # death cross?
                current_signal = 'sell'

            action = 0
            # if current_signal != previous_signal:
            if True:
                if current_signal == 'buy':
                    action = BUY
                elif current_signal == 'sell' and eth_held > 0:
                    action = SELL
            previous_signal = current_signal

        elif strategy == 'MACD':
            current_signal = 'hold'
            if row['MACD'] < row['Signal_Line']:
                current_signal = 'buy'
            elif row['MACD'] > row['Signal_Line']:
                current_signal = 'sell'

            action = 0
            # if current_signal != previous_signal:
            if True:
                if current_signal == 'buy' and cash > 0:
                    action = BUY
                elif current_signal == 'sell' and eth_held > 0:
                    action = SELL
            previous_signal = current_signal

        elif strategy == 'BollingerBands':
            period = sargs['period']  # e.g., 20 for a 20-day SMA
            multiplier = sargs['multiplier']  # Commonly set to 2
            sma = row[f'SMA_{period}']
            sd = row[f'STD_{period}']
            
            upper_band = sma + (sd * multiplier)
            lower_band = sma - (sd * multiplier)

            current_signal = 'hold'
            if open_price < lower_band:
                current_signal = 'buy'
            elif open_price > upper_band:
                current_signal = 'sell'

            action = 0
            # if current_signal != previous_signal:
            if True:
                if current_signal == 'buy' and cash > 0:
                    action = BUY
                elif current_signal == 'sell' and eth_held > 0:
                    action = SELL
            previous_signal = current_signal

        elif strategy == 'buy_and_hold':
            action = 0
            if cash > 0:
                action = FULL_BUY
        
        # # here to add LSTM strategy
        # elif strategy == 'LSTM':
        #     action = lstm_strategy(df, sargs['starting_date'], sargs['ending_date'], look_back=5)
        #     if action == 'Buy' and cash > 0:
        #         action = BUY
        #     elif action == 'Sell' and eth_held > 0:
        #         action = SELL
        #     else:
        #         action = 0

        elif strategy == 'optimal':
            next_open = df_tmp.iloc[index+1]['open']
            if open_price < next_open:
                action = FULL_BUY
            elif open_price > next_open:
                action = FULL_SELL
            else:
                action = 0

        else:
            raise ValueError('Invalid strategy')

        state, reward, done, info = env.step(action)


    net_worth = state['net_worth']
    total_irr = (net_worth / starting_net_worth) - 1
    irrs = np.array(irrs) * 100
    irr_mean = np.mean(irrs)
    irr_std = np.std(irrs)
    risk_free_rate = 0  # same as sociodojo
    result = {
        'total_irr': total_irr,
        'sharp_ratio': (irr_mean - risk_free_rate) / irr_std,
        # add daily return and daily return std
        'daily_return': np.mean(irrs),
        'daily_return_std': np.std(irrs),
    }
    result_str = f'Total IRR: {total_irr*100:.2f} %, Sharp Ratio: {result["sharp_ratio"]:.2f}, Daily Return: {result["daily_return"]:.2f} %, Daily Return Std: {result["daily_return_std"]:.2f} %'
    print(result_str)
    

# strategy = 'LSTM'
# print(strategy)
# run_strategy(strategy, {'starting_date': TEST_START, 'ending_date': TEST_END})


# strategy = 'optimal'
# print(strategy)
# run_strategy(strategy, {'starting_date': TEST_START, 'ending_date': TEST_END})


strategy = 'buy_and_hold'
print(strategy)
sargs = {'starting_date': TEST_START, 'ending_date': TEST_END, 'dataset': 'btc'}
run_strategy(strategy, sargs)


strategy = 'SMA'
# for period in sma_periods:
#     sargs = {'period': period, 'starting_date': VAL_START, 'ending_date': VAL_END}
#     print(f'{strategy}, Period: {period}')
#     run_strategy(strategy, sargs)

period = 15
print(f'{strategy}, Period: {period}')
sargs = {'period': period, 'starting_date': TEST_START, 'ending_date': TEST_END, 'dataset': 'eth'}
run_strategy(strategy, sargs)


strategy = 'SLMA'
for i in range(len(sma_periods)-1):
    for j in range(i+1, len(sma_periods)):
        short = f'SMA_{sma_periods[i]}'
        long = f'SMA_{sma_periods[j]}'
        sargs = {'short': short, 'long': long, 'starting_date': VAL_START, 'ending_date': VAL_END, 'dataset': 'eth'}
        print(f'{strategy}, Short: {short}, Long: {long}')
        run_strategy(strategy, sargs)

short, long = 'SMA_15', 'SMA_30'
sargs = {'short': short, 'long': long, 'starting_date': TEST_START, 'ending_date': TEST_END, 'dataset': 'eth'}
print(f'{strategy}, Short: {short}, Long: {long}')
run_strategy(strategy, sargs)


strategy = 'MACD'
sargs = {'starting_date': TEST_START, 'ending_date': TEST_END, 'dataset': 'eth'}
print(f'{strategy}')
run_strategy(strategy, sargs)


strategy = 'BollingerBands'
period = 20
multiplier = 2
sargs = {'period': period, 'multiplier': multiplier, 'starting_date': TEST_START, 'ending_date': TEST_END, 'dataset': 'eth'}
print(f'{strategy}, Period: {period}, Multiplier: {multiplier}')
run_strategy(strategy, sargs)
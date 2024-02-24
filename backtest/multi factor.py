import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

file_path =  '/Users/jinnia/Desktop/data1/Merged_Hurst_Exponent.csv'

df = pd.read_csv(file_path, parse_dates=['Date'])
df.set_index('Date', inplace=True)

start_date = '2015-01-07'
df = df.loc[start_date:]
df.head()

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

df.fillna(0, inplace=True)

columns_to_calculate = ['WM1NS', 'T10Y2Y', 'WM2NS', 'CPIAUCSL', 'T5YIE']

for column in columns_to_calculate:
    df[f'{column}-WoW'] = df[column].pct_change(periods=1) * 100 # 计算周环比百分比变化
    df[f'{column}-YoW'] = df[column].pct_change(periods=52) * 100

df.fillna(0, inplace=True)

columns_to_calculate = ['Hurst Exponent', 'WM1NS', 'T10Y2Y', 'WM2NS', 'CPIAUCSL', 'T5YIE', 'WM1NS-WoW', 'CPIAUCSL-WoW','WM1NS-YoW', 'CPIAUCSL-YoW']
window_size = 2

for column in columns_to_calculate:
    df[f'{column} rolling mean'] = df[column].rolling(window=window_size).mean().ffill()
    df[f'{column} momentum'] = np.log(df[column] / df[column].shift(6)) - np.log(df[column]/ df[column].shift(1)).ffill()

df


# df['index'] = df.apply(lambda row: 1 if row['WM1NS'] > row['WM1NS rolling mean']  else -1, axis=1)
# df['index'] = df.apply(lambda row: -1 if row['T10Y2Y'] > row['T10Y2Y rolling mean']  else 1, axis=1)
# df['index'] = df.apply(lambda row: -1 if row['WM2NS'] > row['WM2NS rolling mean']  else 1, axis=1)
# df['index'] = df.apply(lambda row: -1 if row['CPIAUCSL'] > row['CPIAUCSL rolling mean']  else 1, axis=1)
# df['index'] = df.apply(lambda row: 1 if row['T5YIE'] > row['T5YIE momentum']  else -1, axis=1)


# 对每个条件创建一个指示列
df['index_Hurst'] = df.apply(lambda row: 1 if row['Hurst Exponent'] > row['Hurst Exponent rolling mean'] else -1, axis=1)
df['index_WM1NS'] = df.apply(lambda row: 2 if row['WM1NS'] > row['WM1NS rolling mean'] else -2, axis=1)
df['index_WM1NS-YoW'] = df.apply(lambda row: 1 if row['WM1NS-YoW'] > row['WM1NS-YoW rolling mean']  else -1, axis=1)
# df['index_T10Y2Y'] = df.apply(lambda row: -1 if row['T10Y2Y'] > row['T10Y2Y rolling mean'] else 1, axis=1)
# df['index_WM2NS'] = df.apply(lambda row: -1 if row['WM2NS'] > row['WM2NS rolling mean'] else 1, axis=1)
# df['index_CPIAUCSL'] = df.apply(lambda row: -1 if row['CPIAUCSL'] > row['CPIAUCSL rolling mean'] else 1, axis=1)
# df['index_T5YIE'] = df.apply(lambda row: 1 if row['T5YIE'] > row['T5YIE momentum'] else -1, axis=1)

# # 计算这些列的平均值以生成一个综合指数
df['index'] = df[['index_Hurst','index_WM1NS','index_WM1NS-YoW']].mean(axis=1).apply(lambda x: 1 if x >= 0 else -1)



sp = yf.download('^GSPC', start='2015-01-07',end='2024-01-10',progress=False)
sp_selected = sp[['Open','High','Low','Close']]

merged_df = df.join(sp_selected, on='Date', how='left')
merged_df = merged_df[['Open', 'High', 'Low', 'Close', 'index']]
merged_df.fillna(method='ffill', inplace=True)  # 前向填充
merged_df.reset_index(inplace=True)
merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')

print(merged_df.head())

merged_df.to_csv('/Users/jinnia/Desktop/backtest/raw data/backtesting_data.csv', index=False)

run /Users/jinnia/Desktop/backtest/cringe.py




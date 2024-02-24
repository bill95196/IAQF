import os
import pandas as pd
import numpy as np
from scipy.stats import linregress

def compute_RS_rolling(series, window_size=30):
    n = len(series)
    RS = []
    dates = []
    
    for start in range(0, n - window_size + 1):
        end = start + window_size
        segment = series[start:end]
        mean = segment.mean()
        cumulative_deviation = np.cumsum(segment - mean)
        R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
        S = segment.std()
        if S != 0:
            RS.append(R / S)
            dates.append(series.index[end-1])
    
    return dates, RS

def hurst_exponent_rolling(series, window_size=30):
    dates, RS = compute_RS_rolling(series, window_size)
    if not RS:
        return pd.DataFrame(columns=['Date', 'Hurst Exponent'])
    

    hurst_values = []
    for rs in RS:
        hurst_values.append(np.log(rs)/np.log(window_size))
    
    results_df = pd.DataFrame({'Date': dates, 'Hurst Exponent': hurst_values})
    return results_df

file_path= '/Users/jinnia/Desktop/data/sp6m_6.csv'
df = pd.read_csv(file_path, parse_dates=['idt'])
df.set_index('idt', inplace=True)

timeseries = df['mu']
results_df = hurst_exponent_rolling(timeseries, 30)
results_df.to_csv('/Users/jinnia/Desktop/data1/Hurst Exponent.csv', index=False)

# rolling mean
window_size = 2
results_df['Rolling Mean Hurst'] = results_df['Hurst Exponent'].rolling(window=window_size).mean()
results_df['Rolling Mean Hurst'] = results_df['Rolling Mean Hurst'].fillna(method='ffill')
results_df['index'] = results_df.apply(lambda row: 1 if row['Hurst Exponent'] > row['Rolling Mean Hurst'] else -1, axis=1)
print(results_df)

# 5-day mometumn
# results_df['Hurst Difference'] = results_df['Hurst Exponent'].shift(1) - results_df['Hurst Exponent'].shift(6)
# results_df['index'] = results_df['Hurst Difference'].apply(lambda x: 1 if x > 0 else -1)
# print(results_df)

sp_df = pd.read_csv('/Users/jinnia/Desktop/data/sp.csv', parse_dates=['Date'])
sp_df.set_index('Date', inplace=True)
sp_selected = sp_df[['Open','High','Low','Close']]
merged_df = results_df.join(sp_selected, on='Date', how='left')
merged_df.dropna(subset=['Open'], inplace=True)
#merged_df = merged_df.drop(['Rolling Mean Hurst', 'Hurst Exponent'], axis=1)
merged_df = merged_df.drop(['Hurst Difference', 'Hurst Exponent'], axis=1)


cols = [col for col in merged_df.columns if col != 'index']
cols.append('index')
merged_df = merged_df[cols]
print(merged_df)

merged_df.to_csv('/Users/jinnia/Desktop/backtest/raw data/backtesting_data.csv', index=False)

run /Users/jinnia/Desktop/backtest/cringe.py
import pandas as pd
import pandas_ta as ta
import numpy as np

def lag_ret(data):
    ret = np.log(data['Close'] / data['Close'].shift(1))
    ret_10 = ret.rolling(window = 10).sum()
    ret_10.name = "lag_ret"
    return ret_10

def lag_vol(data):
    ret = np.log(data['Close'] / data['Close'].shift(1))
    vol_10 = ret.rolling(window = 10).std()
    vol_10.name = "lag_vol"
    return vol_10

def volume_ratio(data):
    volume_ratio = (data['Volume'] - data['Volume'].shift(5))/ data['Volume'].shift(5)
    volume_ratio.replace([np.inf, -np.inf], 0, inplace=True)
    roll_ratio = volume_ratio.rolling(window= 5).mean()
    roll_ratio.name = "volume_ratio"
    return roll_ratio

def relative_strength_index(data):
    rsi = data.ta.rsi().fillna(0)
    return rsi

def relative_volatility_index(data):
    rvi = data.ta.rvi()
    return rvi.fillna(0)

def money_flow_index(data):
    mfi = data.ta.mfi()
    return mfi.fillna(0)

def on_balance_volume(data):
    obv = data.ta.obv()/ 1e10
    return obv.fillna(0)

def self_construct_features(data):
    df = pd.DataFrame()
    df['lag_ret'] = lag_ret(data)
    df['lag_vol'] = lag_vol(data)
    df['volume_ratio'] = volume_ratio(data)
    df['rsi'] = relative_strength_index(data)
    df['rvi'] = relative_volatility_index(data)
    df['mfi'] = money_flow_index(data)
    df['obv'] = on_balance_volume(data)
    
    return df

def additional_data(data):
    dff = pd.read_csv("data/DFF.csv", index_col="DATE") # daily
    dff.index = pd.to_datetime(dff.index)
    
    T10Y2Y = pd.read_csv("data/T10Y2Y.csv", index_col="DATE") # daily
    T10Y2Y.index = pd.to_datetime(T10Y2Y.index)
    T10Y2Y['T10Y2Y'] = pd.to_numeric(T10Y2Y['T10Y2Y'], errors='coerce')
    T10Y2Y.dropna(inplace=True)


    T10YFF = pd.read_csv("data/T10YFF.csv", index_col = "DATE") # daily
    T10YFF.index = pd.to_datetime(T10YFF.index)
    T10YFF['T10YFF'] = pd.to_numeric(T10YFF['T10YFF'], errors='coerce')
    T10YFF.dropna(inplace=True)
    
    ff_all = pd.read_csv("data/ff_all.csv",index_col="date") # daily
    ff_all.index = pd.to_datetime(ff_all.index)
    
    ffund = pd.read_csv("data/FEDFUNDS.csv", index_col="DATE") # monthly
    ffund.index = pd.to_datetime(ffund.index)
    ffund = ffund.resample('D').ffill()
    
    con_senti = pd.read_csv("data/UMCSENT.csv", index_col="DATE") # monthly
    con_senti.index = pd.to_datetime(con_senti.index)
    con_senti = con_senti.resample('D').ffill() 
    
    U2_rate = pd.read_csv("data/U2RATE.csv", index_col="DATE") # monthly
    U2_rate.index = pd.to_datetime(U2_rate.index)
    U2_rate = U2_rate.resample('D').ffill()
    
    hurst = pd.read_csv("data/HurstExponent.csv", index_col="Date")
    hurst.index = pd.to_datetime(hurst.index)
    hurst = hurst.resample('D').ffill()
    
    m1 = pd.read_csv("data/WM1NS.csv", index_col="DATE")
    m1.index = pd.to_datetime(m1.index)
    m1 = m1.resample('D').ffill()
    
    df = pd.DataFrame(index = data.index)
    
    df['dff'] = dff
    df['t10y2y'] = T10YFF
    df['t10yff'] = T10YFF
    df['ffund'] = ffund
    df['con_senti'] = con_senti
    df['u2_rate'] = U2_rate
    df['hurst'] = hurst
    df['m1'] = m1
    
    df = df.merge(ff_all, how = 'left', left_index=True, right_index=True)
    
    return df.ffill()
    

    
    
    
    
    


    







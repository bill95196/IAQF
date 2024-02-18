import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from tools.getData import getData
from tools.set_logger import setupLog

logger = setupLog(ident='TestReturn', level='INFO', handlers_type='console')
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#显示宽度
pd.set_option('display.width', 500)

pairs = {"silver_6":"SI=F",
            "bac_3":"BAC", 
             "citi_3":"C", 
             "corn_6":"ZC=F", 
             "euro_3":"EURUSD=X", 
             "gold_6":"GC=F", 
            #  "iyr_3":"IYR", 
             "oil_6":"CL=F", 
             "pound_3":"GBPUSD=X", 
             "soybns_6":"ZS=F", 
            #  "tr5yr_3":"^FVX", 
            #  "tr10yr_6":"^TNX", 
             "wheat_6":"ZW=F", 
             "yen_3":"JPY=X",
             "sp6m_6":"^GSPC",
             "sp12m_12":"^GSPC"}

for ticker in pairs.keys():
    rnd, yf_price = getData(ticker)

    yf_price['return'] = np.log(yf_price['Close']/yf_price['Close'].shift(1)) # daily return 
    yf_price['return'] = yf_price['return'].shift(-1)
    
    R2 = pd.DataFrame()
    Pvalue = pd.DataFrame()
    time_interval = [10, 30, 50, 80, 110, 130, 150, 180]
    for win in time_interval:
        r2_series = pd.Series(dtype=float)
        qval_series =  pd.Series(dtype=float)
        
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=win) # win size forward return
        yf_price['forward_ret'] = yf_price['return'].rolling(window=indexer).sum()
        rnd['forward_ret'] = yf_price['forward_ret']
        
        selected = ['mu', 'sd', 'skew','kurt','p10','p50','p90','prDec','prInc','forward_ret']
        full_df = rnd[selected].copy().dropna()
        y = full_df['forward_ret']
        full_df = full_df.drop('forward_ret', axis= 1)
        logger.info(f'successfully appending to a full dataframe for time interval {win} days for {ticker}')
        
        for fname in full_df.columns:
            
            X = full_df[fname]
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            
            r_squared = model.rsquared
            r2_series[fname] = round(r_squared,4)
            
            p_values = model.pvalues[fname]
            qval_series[fname] = round(p_values,4)
        
        logger.info(f'window size: {win} ----LR fitting Done for {ticker} ')
        R2[f'{win}dayR2'] = r2_series
        Pvalue[f'{win}dayPvalues'] = qval_series
        
    R2 = R2.T
    Pvalue = Pvalue.T
    combined_df = pd.concat([R2, Pvalue], axis=0)
    combined_df.to_csv(f'Return_info/{ticker}_return.csv')
    logger.info(f'Saved {ticker} results in the folder')

        
        
        
        
        
        


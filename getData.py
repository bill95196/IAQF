import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import yfinance as yf

def name_ticker_pair():
    pairs = {"silver_6":"SI=F", 
             "bac_3":"BAC", 
             "citi_3":"C", 
             "corn_6":"ZC=F", 
             "euro_3":"EURUSD=X", 
             "gold_6":"GC=F", 
             "iyr_3":"IYR", 
             "oil_6":"CL=F", 
             "pound_3":"GBPUSD=X", 
             "soybns_6":"ZS=F", 
             "tr5yr_3":"^FVX", 
             "tr10yr_6":"^TNX", 
             "wheat_6":"ZW=F", 
             "yen_3":"JPY=X",
             "sp6m_6":"^GSPC",
             "sp12m_12":"^GSPC"}
    return pairs

def getRnd(Ticker:str):
    filename = 'data/' + Ticker + '.csv'
    
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['idt'], format = '%m/%d/%y')
    data = data.set_index('Date')
    data.drop('idt', axis=1, inplace=True)
    
    return data

def getDailyPrice(Ticker: str):
    pairs = name_ticker_pair()
    yf_Ticker = pairs[Ticker]
    data = yf.download(yf_Ticker, start = '2007-01-12', end = '2024-01-22', progress=False)
    return data

if __name__ == "__main__":
    ticker = "sp6m_6"
    sp_rnd = getRnd(ticker)
    sp_yf = getDailyPrice(ticker)

    print(sp_rnd)


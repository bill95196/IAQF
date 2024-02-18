import pandas as pd
import yfinance as yf
from tools.set_logger import setupLog

logger = setupLog(ident='TestReturn', level='INFO', handlers_type='console')

def name_ticker_pair():
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
    return pairs

# deprecated
def getRnd(Ticker:str):
    filename = 'data/' + Ticker + '.csv'
    
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['idt'], format = '%m/%d/%y')
        
    data = data.set_index('Date')
    data.drop('idt', axis=1, inplace=True)
    
    return data


# deprecated
def getDailyPrice(Ticker: str):
    
    rnd_df = getRnd(Ticker)
    ts = rnd_df.index.to_series()
    rnd_start = ts.iloc[0]
    rnd_end = ts.iloc[-1]
    
    start_date = (rnd_start - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
    end_date = (rnd_end + pd.Timedelta(days=20)).strftime('%Y-%m-%d')   
    
    pairs = name_ticker_pair()
    yf_Ticker = pairs[Ticker]
    data = yf.download(yf_Ticker, start = start_date, end = end_date, progress=False)
    return data

def getData(Ticker: str):
    filename = 'data/' + Ticker + '.csv'
    
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['idt'], format = '%m/%d/%y')
    
    rnd_start = data['Date'].iloc[0]
    rnd_end = data['Date'].iloc[-1]
    
    data = data.set_index('Date')
    data.drop('idt', axis=1, inplace=True)  
    logger.info(f'successfully load Risk Neutral Probability dataset for {Ticker}\n')
    
    start_date = (rnd_start - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
    end_date = (rnd_end + pd.Timedelta(days=20)).strftime('%Y-%m-%d') 
    
    pairs = name_ticker_pair()
    yf_Ticker = pairs[Ticker]
    yfdata = yf.download(yf_Ticker, start = start_date, end = end_date, progress=False)
    logger.info(f'successfully fetch data from yahoo finance for {Ticker}\n')
    return data, yfdata  
      

if __name__ == "__main__":
    ticker = "citi_3"

    print(getData(ticker))


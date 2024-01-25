from statsmodels.sandbox import distributions
import pandas as pd
import numpy  as np
import yfinance as yf
from utils import *
import scipy.stats as ss
import os
from non_normal import fleishman



data = pd.read_csv('mpd_stats.csv')
data.head()

sets = data.groupby(['market']).indices
for i in sets:
    maturity = data.loc[data['market']==i].groupby(['maturity_target']).indices
    for m in maturity:
        try:
            rnp = pd.read_csv(f'data/{i}_{int(m)}.csv', index_col=0)
            m_data = pd.read_csv(f'data/{i}.csv')
        except:
            print(f'There is no data for {i}')
            continue

        for j in range(rnp.shape[0]):
            sliced_data = rnp.iloc[j]
            mean = sliced_data['mu']
            std = sliced_data['sd']
            skew = sliced_data['skew']
            kurt = sliced_data['kurt']
            
            try:
                coeff = fit_fleishman_from_sk(skew, kurt)
                sim = (generate_fleishman(-coeff[1],*coeff,N=100000))*std+mean
            except:
                print('Error occur at: ', i, m, mean, std, skew, kurt, kurt - (-1.13168 + 1.58837 * skew**2))

                continue
            
            # print(i, m, mean, std**2, skew, kurt+3)
            # print(i, m, describe(sim))

        
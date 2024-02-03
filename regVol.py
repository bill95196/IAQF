import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from getData import getRnd, getDailyPrice

ticker = "gold_6"
rnd = getRnd(ticker)
yf_price = getDailyPrice(ticker)

    
yf_price['return'] = np.log(yf_price['Close']/yf_price['Close'].shift(1)) # daily return 

win = 180
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=win) # win size forward realized vol
yf_price['forward_rv'] = np.sqrt((yf_price['return']**2).rolling(window=indexer, min_periods=1).sum())

rnd['forward_vol'] = yf_price['forward_rv']

# generate plot
fig, axes = plt.subplots(2,1,figsize=(10, 6))
axes[0].plot(rnd.index,rnd['forward_vol'], label = f'future {win} day rv')
axes[1].plot(rnd.index,rnd['sd'], label = 'rnd std')

axes[0].legend()
axes[1].legend()
axes[0].set_ylabel(f'volatility')
axes[1].set_ylabel(f'volatility')
fig.suptitle(f"{ticker}'s volatility with time interval {win} days",fontsize=16)
plt.xlabel('Time')
plt.savefig(f'vol_figure/{ticker}@{win}days.png')

# run simple OLS for forward rv on rnd std
import statsmodels.api as sm
reg_df = rnd[['sd','forward_vol']].copy().dropna()
X = reg_df["sd"]
y = reg_df["forward_vol"]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
coefficients = model.params
t_values = model.tvalues
p_values = model.pvalues
r_squared = model.rsquared

stats_df = pd.DataFrame({
    'Coefficient': coefficients,
    'T-value': t_values,
    'P-value': p_values
})

# Add R-squared value as a new row or separate entry
stats_df.loc['R-squared'] = [r_squared, None, None]  # R-squared does not have t-value and p-value
stats_df.to_csv(f'vol_reg_output/{ticker}@{win}days.csv')




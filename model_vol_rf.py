import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix,mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight

from tools.training_tools import generate_feature_importances_plot, generate_predict_plot
from tools.getData import getData
from tools.set_logger import setupLog
logger = setupLog(ident='rf_vol', level='INFO', handlers_type='console')

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#显示宽度
pd.set_option('display.width', 500)


ticker = 'sp6m_6'
rnd_data, dailyprice = getData(ticker)

# preprocess data: create label and merge to rnd dataset
log_return = np.log(dailyprice['Close']/dailyprice['Close'].shift(1))
log_return = log_return.shift(-1)

indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=15) # win size forward volatility
real_vol = np.sqrt((log_return**2).rolling(window=indexer).sum())

rnd_data['label'] = real_vol
rnd_data.drop(['market','maturity_target','lg_change_decr','lg_change_incr'], axis = 1, inplace= True)
rnd_data.dropna(inplace=True)

# overall data distribution
logger.info(f'feature/label: \n {rnd_data.describe().T}')





def train_test_split(features: pd.DataFrame, split_ratio = 0.7):
    n = int(len(features) * split_ratio)
    x = features.iloc[:n, :-1]
    y = features.iloc[:n, -1]
    x_test = features.iloc[n:, :-1]
    y_test = features.iloc[n:, -1]
    
    return x, y, x_test, y_test
    
def grid_search(x_train, y_train):
    
    rf = RandomForestRegressor()
    
    param_grid = {
    'n_estimators': [10,20, 30, 40, 50],
    'criterion' : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    'max_depth': [3, 5, 7, 10, 15, 20],
    'min_samples_leaf': [3, 5, 10, 15,20, 30, 40, 50],
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=0)
    grid_search.fit(x_train, y_train)
    logger.info(f'\n Best paramters: {grid_search.best_params_} \n')
    
    return grid_search.best_params_

 
def train_rf(model_params, x_train, y_train, x_test, y_test):
    logger.info(f'shape of x: {x_train.shape} \n ')
    
    rf = RandomForestRegressor(n_estimators=model_params['n_estimators'],
                                    criterion=model_params['criterion'],
                                    max_depth=model_params['max_depth'], 
                                    min_samples_leaf=model_params['min_samples_leaf'], 
                                    random_state=10)
    
    rf.fit(x_train, y_train)
    
    pred = rf.predict(x_test)
    real = y_test
    pred = pd.Series(pred, index = real.index)
    
    mse = mean_squared_error(real, pred)
    logger.info(f"Mean Squared Error (MSE): {mse}")

    # Calculating R-squared
    r2 = r2_score(real, pred)
    logger.info(f"R2: {r2}")

    _ = generate_feature_importances_plot(rf, x_test)
    _ = generate_predict_plot(rf, x_train = x, y_train = y, x_test = x_test, y_test = y_test)
    
    

    
x, y, x_test, y_test = train_test_split(features= rnd_data, split_ratio= 0.7)
best_params = grid_search(x, y)
train_rf(model_params= best_params, x_train= x, y_train= y, x_test= x_test, y_test= y_test)



    
    
    
    
    
    
    
    
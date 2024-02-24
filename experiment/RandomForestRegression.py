import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from feature_engineering import features, more_data
from tools.getData import getData
from tools.training_tools import generate_predict_plot, confusion_matrix_reg, feature_importances_plot

from tools.set_logger import logger
import warnings
warnings.filterwarnings("ignore")

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#显示宽度
pd.set_option('display.width', 500)



ticker = "sp6m_6"
rnd_data, dailyprice = getData(ticker)

# add additional features
feature_df = features(dailyprice)
add_data = more_data(dailyprice)
rnd_data = rnd_data.merge(feature_df, how='left', left_index=True, right_index=True)
rnd_data = rnd_data.merge(add_data, how='left', left_index=True, right_index=True)



rnd_data["Close_Price"] = dailyprice["Close"]
rnd_data["label"] = (rnd_data["Close_Price"].shift(-1) - rnd_data["Close_Price"]) / rnd_data["Close_Price"]


rnd_data.drop(["market","maturity_target","lg_change_decr","lg_change_incr", "Close_Price"], axis = 1, inplace= True)
rnd_data.dropna(inplace=True)

# overall data distribution
logger.info(f'feature/label: \n {rnd_data.describe().T}')


def train_test_split(features: pd.DataFrame, train_start, train_end):
    train_start = pd.to_datetime(train_start, format='%Y-%m-%d')
    train_end = pd.to_datetime(train_end, format='%Y-%m-%d')
    
    x_train = features.iloc[(features.index >= train_start) & (features.index <= train_end) , :-1]
    y_train = features.iloc[(features.index >= train_start) & (features.index <= train_end), -1]
    x_test = features.iloc[features.index > train_end, :-1]
    y_test = features.iloc[features.index > train_end, -1]
    
    return x_train, y_train, x_test, y_test


def grid_search(x_train, y_train):
    
    rf = RandomForestRegressor()
    
    param_grid = {
    "n_estimators": [30, 50, 60, 70, 100, 200],
    "criterion" : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "max_depth": [3, 5, 10, 15, 20, 30, 40, 50],
    "min_samples_leaf": [3, 5, 10, 15, 20, 30, 40, 50]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=0)
    grid_search.fit(x_train, y_train)
    logger.info(f'\n Best paramters: {grid_search.best_params_} \n')
    logger.info(f"\n Best score: {grid_search.best_score_} \n")
    
    return grid_search.best_params_

 
def train_rf(model_params, x_train, y_train, x_test, y_test):
    logger.info(f'shape of x: {x_train.shape} \n ')
    
    rf = RandomForestRegressor(n_estimators=model_params["n_estimators"],
                               criterion=model_params["criterion"],
                               max_depth=model_params["max_depth"], 
                               min_samples_leaf=model_params["min_samples_leaf"], 
                               random_state=10)
    
    rf.fit(x_train, y_train)

    pred = rf.predict(x_test)
    real = y_test
    pred = pd.Series(pred, index = real.index, name = "pred")
    
    # plot
    generate_predict_plot(rf, x_train, y_train, x_test, y_test)
    feature_importances_plot(rf, x_test)
    
    # confusion matrix
    confusion_matrix_reg(real, pred)
    
    pred_class = pred.apply(lambda x: 1 if x > 0 else -1)
    return pred_class
    

x, y, x_test, y_test = train_test_split(features= rnd_data, 
                                        train_start= "2020-01-01", 
                                        train_end = "2022-01-01")

best_params = grid_search(x, y)

yhat = train_rf(model_params= best_params, 
                x_train= x, 
                y_train= y, 
                x_test= x_test, 
                y_test= y_test)
yhat.name = "signals"

df = pd.merge(yhat,dailyprice,how='left',left_index=True, right_index=True)
df['index'] = yhat
df.drop(['signals','Adj Close', 'Volume'], axis=1, inplace=True)
logger.info(f'first 5 row: \n {df.head(5)}')

df.to_csv('backtest.csv')
    
    
    
    
    
    
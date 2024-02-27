import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
import warnings; warnings.filterwarnings("ignore")

from tools.set_logger import logger
from tools.getData import getData
from tools.training_tools import confusion_matrix_clf, feature_importances_plot
from feature_engineering import self_construct_features, additional_data

#show all the columns
pd.set_option('display.max_columns', None)
#show all the columns
pd.set_option('display.max_rows', None)
#set width
pd.set_option('display.width', 500)

ticker = 'sp6m_6'
rnd_data, dailyprice = getData(ticker)

# add additional features
feature_df = self_construct_features(dailyprice)
add_data = additional_data(dailyprice)
rnd_data = rnd_data.merge(feature_df, how='left', left_index=True, right_index=True)
rnd_data = rnd_data.merge(add_data, how='left', left_index=True, right_index=True)

# define the label
rnd_data["Close_Price"] = dailyprice["Close"]
rnd_data["return"] = (rnd_data["Close_Price"].shift(-1) - rnd_data["Close_Price"]) / rnd_data["Close_Price"]
for_test = rnd_data.copy()

rnd_data['label'] = rnd_data['return'].apply(lambda x: 1 if x > 0 else -1)
rnd_data.drop(['market','maturity_target','lg_change_decr','lg_change_incr','return', 'Close_Price'], axis = 1, inplace= True)
rnd_data.dropna(inplace=True)

# overall data distribution
logger.info(f'feature/label: \n {rnd_data.describe().T}')

def train_test_split(features: pd.DataFrame, train_start: str, train_end: str):
    train_start = pd.to_datetime(train_start, format='%Y-%m-%d')
    train_end = pd.to_datetime(train_end, format='%Y-%m-%d')
    
    x_train = features.iloc[(features.index >= train_start) & (features.index <= train_end) , :-1]
    y_train = features.iloc[(features.index >= train_start) & (features.index <= train_end), -1]
    x_test = features.iloc[features.index > train_end, :-1]
    y_test = features.iloc[features.index > train_end, -1]
    return x_train, y_train, x_test, y_test
    
def grid_search(x_train, y_train):
    
    rf = RandomForestClassifier(random_state=10)
    
    param_grid = {
    'n_estimators': [10, 15, 20, 30, 40, 50, 60],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_leaf': [10, 15, 20, 30, 40, 50],
    'max_samples': [0.4, 0.5, 0.6, 0.7, 0.8],
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=0)
    grid_search.fit(x_train, y_train)
    logger.info(f'\n Best paramters: {grid_search.best_params_} \n')
    logger.info(f"\n Best score: {grid_search.best_score_} \n")
    return grid_search.best_params_

def train_rf(model_params, x_train, y_train, x_test, y_test):
    logger.info(f'shape of x: {x_train.shape} \n ')
    
    class_labels, counts = np.unique(y_train.values, return_counts=True)
    logger.info(f'class disturbution: {dict(zip(class_labels, counts))}')
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
    class_weight_dict = {class_labels[i]: class_weights[i] for i in range(len(class_labels))}
    
    rf = RandomForestClassifier(n_estimators=model_params['n_estimators'],
                                max_depth=model_params['max_depth'],
                                min_samples_leaf=model_params['min_samples_leaf'], 
                                max_samples=model_params['max_samples'], 
                                random_state=10,
                                class_weight= class_weight_dict)
    rf.fit(x_train, y_train)

    pred = rf.predict(x_test)
    real = y_test
    pred = pd.Series(pred, index = real.index, name = 'signals')

    # confusion matrix
    confusion_matrix_clf(real, pred)
    # draw feature importance plot
    feature_importances_plot(rf, x_test)

    return pred
    

x, y, x_test, y_test = train_test_split(features= rnd_data, 
                                        train_start= "2020-01-01", 
                                        train_end = "2022-01-01")

best_params = grid_search(x, y)

yhat = train_rf(model_params= best_params, 
                x_train= x, 
                y_train= y, 
                x_test= x_test, 
                y_test= y_test)

df = pd.merge(yhat,dailyprice,how='left',left_index=True, right_index=True)
df['index'] = yhat
df.drop(['signals','Adj Close', 'Volume'], axis=1, inplace=True)
logger.info(f'first 5 row: \n {df.head(5)}')

df.to_csv('backtest.csv')



x, y, x_test, y_test = train_test_split(features= for_test, 
                                        train_start= "2020-01-01", 
                                        train_end = "2022-01-01")

new_df = pd.DataFrame()
new_df["return"] = y_test
new_df["pred"] = yhat

n = pd.Series([len(yhat[yhat == -1]), len(yhat[yhat == 1])], index = [-1, 1])
reslt_mean = new_df.groupby('pred').mean()['return'].reindex([-1, 1])
reslt_std = new_df.groupby('pred').std()['return'].reindex([-1, 1])

output = pd.DataFrame({"num": n, "mean": reslt_mean, "std": reslt_std}, index = [-1, 1])
output['std_err'] = output['std'] / np.sqrt(output['num'])
output['t'] = output['mean'] / output['std_err']
logger.info(f"{output}")
    
    
    
    
    
    
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from feature_engineering import features, more_data
from tools.getData import getData
from tools.training_tools import confusion_matrix_clf, feature_importances_plot

from tools.set_logger import logger
import warnings
warnings.filterwarnings("ignore")

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#显示宽度
pd.set_option('display.width', 500)



ticker = 'sp6m_6'
rnd_data, dailyprice = getData(ticker)

# add additional features
feature_df = features(dailyprice)
add_data = more_data(dailyprice)
rnd_data = rnd_data.merge(feature_df, how='left', left_index=True, right_index=True)
rnd_data = rnd_data.merge(add_data, how='left', left_index=True, right_index=True)

# create label and merge to rnd dataset
return_15 = (dailyprice['Close'].shift(-15) - dailyprice['Close']) / dailyprice['Close']
rnd_data['ret_15'] = return_15

rnd_data['label'] = rnd_data['ret_15'].apply(lambda x: 1 if x > 0 else -1)
rnd_data.drop(['market','maturity_target','lg_change_decr','lg_change_incr','ret_15'], axis = 1, inplace= True)
rnd_data.dropna(inplace=True)

# overall data distribution
logger.info(f'feature/label: \n {rnd_data.describe().T}')

# examine for balanced label
unique, counts = np.unique(rnd_data['label'].values, return_counts=True)
logger.info(f'class disturbution: {dict(zip(unique, counts))}')

# label inbalance issue solving
class_labels = np.unique(rnd_data['label'])
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=rnd_data['label'])
class_weight_dict = {class_labels[i]: class_weights[i] for i in range(len(class_labels))}
logger.info(f'class weight: {class_weight_dict}')


def train_test_split(features: pd.DataFrame, split_ratio = 0.7):
    n = int(len(features) * split_ratio)
    x = features.iloc[:n, :-1]
    y = features.iloc[:n, -1]
    x_test = features.iloc[n:, :-1]
    y_test = features.iloc[n:, -1]
    return x, y, x_test, y_test
    
def grid_search(x_train, y_train):
    
    rf = RandomForestClassifier()
    
    param_grid = {
    'n_estimators': [10, 20, 30, 40, 50, 60],
    'max_depth': [3, 5, 8, 10, 15, 20, 30],
    'min_samples_leaf': [3, 5, 10, 15, 20, 30, 40, 50],
    'max_samples': [0.4, 0.5, 0.6, 0.7, 0.8]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=0)
    grid_search.fit(x_train, y_train)
    logger.info(f'\n Best paramters: {grid_search.best_params_} \n')
    logger.info(f"\n Best score: {grid_search.best_score_} \n")
    return grid_search.best_params_

 
def train_rf(model_params, x_train, y_train, x_test, y_test):
    logger.info(f'shape of x: {x_train.shape} \n ')
    
    rf = RandomForestClassifier(n_estimators=model_params['n_estimators'],
                                max_depth=model_params['max_depth'],
                                min_samples_leaf=model_params['min_samples_leaf'], 
                                max_samples=model_params['max_samples'], 
                                random_state=10,
                                class_weight=class_weight_dict)
    rf.fit(x_train, y_train)

    pred = rf.predict(x_test)
    real = y_test
    pred = pd.Series(pred, index = real.index, name = 'signals')

    # confusion matrix
    confusion_matrix_clf(real, pred)
    # draw feature importance plot
    feature_importances_plot(rf, x_test)

    return pred
    

x, y, x_test, y_test = train_test_split(features= rnd_data, split_ratio= 0.7)
best_params = grid_search(x, y)
yhat = train_rf(model_params=best_params, 
                x_train= x, 
                y_train= y, 
                x_test= x_test, 
                y_test= y_test)

df = pd.merge(yhat,dailyprice,how='left',left_index=True, right_index=True)
df['index'] = yhat
df.drop(['signals','Adj Close', 'Volume'], axis=1, inplace=True)
logger.info(f'first 5 row: \n {df.head(5)}')

# df.to_csv('backtest.csv')
    
    
    
    
    
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix

from tools.getData import getData
from tools.set_logger import setupLog
logger = setupLog(ident='rf_return', level='INFO', handlers_type='file')

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#显示宽度
pd.set_option('display.width', 500)



ticker = 'bac_3'
rnd_data, dailyprice = getData(ticker)

# preprocess data: create label and merge to rnd dataset
return_7 = (dailyprice['Close'].shift(-15) - dailyprice['Close']) / dailyprice['Close']
rnd_data['ret_7'] = return_7

rnd_data['label'] = rnd_data['ret_7'].apply(lambda x: 1 if x > 0 else 0)
rnd_data.drop(['market','maturity_target','lg_change_decr','lg_change_incr','ret_7'], axis = 1, inplace= True)
rnd_data.dropna(inplace=True)

# overall data distribution
logger.info(f'feature/label: \n {rnd_data.describe().T}')

# examine for balanced label
unique, counts = np.unique(rnd_data['label'].values, return_counts=True)
logger.info(f'class disturbution: \n {dict(zip(unique, counts))}')


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
    'n_estimators': [10,20, 30, 40, 50],
    'max_depth': [3, 5, 7, 10, 15, 20],
    'min_samples_leaf': [3, 5, 10, 15,20, 30, 40, 50],
    'max_samples': [0.3, 0.4, 0.5, 0.6]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=0)
    grid_search.fit(x_train, y_train)
    logger.info(f'\n Best paramters: {grid_search.best_params_} \n')
    
    return grid_search.best_params_


def feature_importance_plot(model, x_test):
    importances = pd.Series(model.feature_importances_, index=x_test.columns)
    importances.sort_values(ascending=True, inplace=True)
    importances.plot.barh(color='green')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()
    
def train_rf(model_params, x_train, y_train, x_test, y_test):
    logger.info(f'shape of x: {x_train.shape} \n ')
    
    rf = RandomForestClassifier(n_estimators=model_params['n_estimators'],
                                    max_depth=model_params['max_depth'],
                                    min_samples_leaf=model_params['min_samples_leaf'], 
                                    max_samples=model_params['max_samples'], 
                                    random_state=10)
    rf.fit(x_train, y_train)
    
    pred = rf.predict(x_test)
    real = y_test
    pred = pd.Series(pred, index = real.index)

    # confusion matrix
    cm = confusion_matrix(real, pred)
    logger.info(f'confusion matrix: \n {cm}')
    
    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    logger.info(f'precision: {precision:.3f}')
    logger.info(f'recall: {recall:.3f}')
    logger.info(f'accuracy: {accuracy:.3f}')
    
    # draw feature importance plot
    importances = pd.Series(rf.feature_importances_, index=x_test.columns)
    importances.sort_values(ascending=True, inplace=True)
    importances.plot.barh(color='green')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()
    
    
x, y, x_test, y_test = train_test_split(features= rnd_data, split_ratio= 0.7)
best_params = grid_search(x, y)
train_rf(model_params= best_params, x_train= x, y_train= y, x_test= x_test, y_test= y_test)


    
    
    
    
    
    
    
    
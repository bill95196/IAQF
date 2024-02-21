import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tools.set_logger import logger

def confusion_matrix_reg(pred: pd.Series, real: pd.Series, thred=0.001):
    real_class = real.apply(lambda x: 1 if x >= thred else 0)
    pred_class = pred.apply(lambda x: 1 if x >= thred else 0)
    
    data = pd.DataFrame({"pred" : pred_class, "real" : real_class})
    rslt = pd.DataFrame()
    for ipred in [0, 1]:
        data_p = data.loc[data["pred"] == ipred]
        r = data_p.groupby("real").count()
        rslt[f"p{ipred}"] = r.reindex([0, 1])
    rslt = rslt.T
    rslt.loc["total_pred"] = rslt.sum(0)
    rslt["total_real"] = rslt.sum(1)
    logger.info(f"confusion_matrix:\n {rslt}")
    logger.info(f"precision:\n {rslt}")
    precision = rslt.loc["p1", 1] / (rslt.loc["p1", 0] + rslt.loc["p1", 1])
    logger.info(f"precision:\n {precision}")
    
def confusion_matrix_clf(true_value: pd.Series, prediction:pd.Series):
    cm = confusion_matrix(true_value, prediction)
    logger.info(f'confusion matrix:\n {cm}')
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    logger.info(f'precision: {precision:.3f}')
    logger.info(f'recall: {recall:.3f}')
    logger.info(f'accuracy: {accuracy:.3f}')

def feature_importances_plot(model, test_series):
    importances = pd.Series(model.feature_importances_, index=test_series.columns)
    importances.sort_values(ascending=True, inplace=True)
    importances.plot.barh(color='green')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()
    
def generate_predict_plot(model, x_train, y_train, x_test, y_test):
    y_in_sample = pd.Series(model.predict(x_train), index = y_train.index)
    y_out_sample =pd.Series(model.predict(x_test), index = y_test.index)
    
    last_timestamp = y_in_sample.index.to_list()[-1]
    
    full_pred = pd.concat([y_in_sample, y_out_sample], axis= 0)
    full_real = pd.concat([y_train, y_test], axis= 0)
    
    plt.plot(full_real.index, full_real, label = 'real')
    plt.plot(full_real.index, full_pred,label = 'pred')
    plt.axvline(last_timestamp, color='r', linestyle='--',linewidth=0.5)
    plt.legend()
    plt.show()
    return None




    
    
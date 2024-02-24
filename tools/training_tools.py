import pandas as pd
import matplotlib.pyplot as plt
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
    
def confusion_matrix_reg(true_value: pd.Series, prediction:pd.Series):
    real_class = true_value.apply(lambda x: 1 if x > 0 else -1)
    pred_class = prediction.apply(lambda x: 1 if x > 0 else -1)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(real_class, pred_class)
    logger.info(f'confusion matrix:\n {cm}')
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    logger.info(f'precision: {precision:.3f}')
    logger.info(f'recall: {recall:.3f}')
    logger.info(f'accuracy: {accuracy:.3f}')

    
def confusion_matrix_clf(true_value: pd.Series, prediction:pd.Series):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_value, prediction)
    logger.info(f'confusion matrix:\n {cm}')
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    logger.info(f'precision: {precision:.3f}')
    logger.info(f'recall: {recall:.3f}')
    logger.info(f'accuracy: {accuracy:.3f}')

def feature_importances_plot(model, x_test):
    importances = pd.Series(model.feature_importances_, index=x_test.columns)
    importances.sort_values(ascending=True, inplace=True)
    importances.plot.barh(color='blue')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()
    
def generate_predict_plot(model, x_train, y_train, x_test, y_test):
    y_in_sample = pd.Series(model.predict(x_train), index = y_train.index)
    y_out_sample =pd.Series(model.predict(x_test), index = y_test.index)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # In-sample fitting plot
    axs[0].plot(x_train.index, y_train, label="Actual", color="blue")
    axs[0].plot(x_train.index, y_in_sample, label="Predicted", color="orange")
    axs[0].set_title("In-Sample Fitting")
    axs[0].legend()

    # Out-of-sample prediction plot
    axs[1].plot(x_test.index, y_test, label="Actual", color="blue")
    axs[1].plot(x_test.index, y_out_sample, label="Predicted", color="orange")
    axs[1].set_title('Out-Of-Sample Prediction')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    





    
    
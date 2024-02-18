import pandas as pd
import matplotlib.pyplot as plt
    
    
def generate_feature_importances_plot(model, x_test):
    # draw feature importance plot
    importances = pd.Series(model.feature_importances_, index=x_test.columns)
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
    
    
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from score_model import rmsle

def cross_val_scores(model, X_data, y_data, num_folds=3):
    ''' Returns error for k-fold cross validation. '''
    kf = KFold(n_splits=num_folds)
    train_error = np.empty(num_folds)
    test_error = np.empty(num_folds)
    index = 0
    for train, test in kf.split(X_data):
        model.fit(X_data[train], y_data[train])
        pred_train = model.predict(X_data[train])
        pred_test = model.predict(X_data[test])
        train_error[index] = rmsle(y_data[train], pred_train)
        test_error[index] = rmsle(y_data[test], pred_test)
        index += 1
    return np.mean(test_error), np.mean(train_error)

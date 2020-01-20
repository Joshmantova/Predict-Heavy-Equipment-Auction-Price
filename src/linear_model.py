import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from clean_data import hotcode_productgroup, hotcode_enclosure, hotcode_state, hotcode_modelid
from sklearn.model_selection import train_test_split
from kfold_func import cross_val_scores
from score_model import rmsle
# from imputation import knn_reg_impute


def fit_sm(model, X, y, reg=False):
    
    if reg:
        model(y, X).fit_regularized()
    else:
        model(y, X).fit()
    
    return model

def fit_sk(model, X, y):
    
    model.fit(X, y)
    
    return model

def lin_reg_sk(X, y, reg=None):
    
    if reg == 'ridge':
        model = Ridge()
    elif reg == 'lasso':
        model = Lasso()
    else:
        model = LinearRegression()
        
    fit_sk(model, X, y)
    
    return model

def log_reg_sk(X, y, reg=None):

    if reg == 'ridge':
        model = LogisticRegression(penalty='l2')
    elif reg == 'lasso':
        model = LogisticRegression(penalty='l1')
    else:
        model = LogisticRegression(penalty='none')
        
    fit_sk(model, X, y)
    
    return model

def create_design_matrix(X):
        return np.hstack([
        hotcode_modelid(X).values, 
        hotcode_enclosure(X).values, 
        hotcode_state(X).values])
        
if __name__ == '__main__':
    
    df_train = pd.read_csv('../data/Train.csv')
    # df_test = pd.read_csv('../data/test.csv')
    # df_holdout = pd.read_csv('../data/end_of_day/test_actual.csv')
    
    X = df_train.drop('SalePrice', axis=1)
    y = df_train['SalePrice']
    
    # X_hold = df_test.values
    # y_hold = df_holdout.values
    
    X_sub = X.iloc[:1000, :]
    
    y_sub = y.iloc[:1000].values
    
    X_sub = create_design_matrix(X_sub)

    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub)
    
    lin_model = lin_reg_sk(X_train, y_train, 'ridge')
    y_pred_lin = lin_model.predict(X_test)
    print(rmsle(y_test, y_pred_lin))
    print(cross_val_scores(X_sub, y_sub, 5))
    
    log_model = log_reg_sk(X_train, y_train, 'ridge')
    y_pred_log = log_model.predict(X_test)
    print(rmsle(y_test, y_pred_log))
    print(cross_val_scores(X_sub, y_sub, 5))

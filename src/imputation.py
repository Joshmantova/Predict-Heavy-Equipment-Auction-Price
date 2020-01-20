import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

def knn_reg_impute(df, x_cols, y_col, index_col):
    X_null = df[df[y_col].isna()][x_cols[:-1]].copy()
    X_null.set_index(index_col, inplace=True)

    X_train = df[df[y_col].notna()][x_cols].copy()
    X_train.set_index(index_col, inplace=True)

    y_train = X_train[[y_col]].copy()

    X_train.drop(y_col, axis=1, inplace=True)

    neigh = KNeighborsRegressor()
    neigh.fit(X_train, y_train)
    predictions = neigh.predict(X_null)
    X_null[y_col] = predictions
    return X_null

if __name__ == '__main__':
    df = pd.read_csv('data/Train.csv')
    x_cols = ['SalePrice', 'YearMade', 'SalesID', 'MachineHoursCurrentMeter']
    y_col = 'MachineHoursCurrentMeter'
    index_col = 'SalesID'
    predictions = knn_reg_impute(df, x_cols, y_col, index_col)
    print(predictions)

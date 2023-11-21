import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



def loadData(filename):
    # Loading the data
    data = pd.read_csv(filename, delimiter=',')

    # Transforming features using log(x_ij + 0.1)
    X = np.log(data.iloc[:, :-1].values + 0.1)
    y = data.iloc[:, -1].values
    
    # Splitting the dataset according to the predefined sizes
    X_train = X[:3065, :]
    y_train = (y[:3065]).reshape(-1, 1)
    X_test = X[3065:, :]
    y_test = (y[3065:]).reshape(-1, 1)

    # Transforming labels from {0, 1} to {-1, 1}
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = loadData('spambase.data')
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
   
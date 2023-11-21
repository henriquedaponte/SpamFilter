import numpy as np
import pandas as pd



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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, weights, lr, iterations):
    for _ in range(iterations):
        predictions = (sigmoid(np.dot(X, weights))).reshape(-1, 1)
        weights = weights.reshape(-1, 1)
        weights -= lr * np.dot(X.T, predictions - y) / len(X)

    return weights

def logistic_regression(X, y, lr, iterations):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    weights = np.zeros(X.shape[1])
    weights = gradient_descent(X, y, weights, lr, iterations)
    return weights

def evaluationMetrics(y_pred, y_test):

    truePositives = np.sum(np.where((y_pred == 1) & (y_test == 1), 1, 0))
    falsePositives = np.sum(np.where((y_pred == 1) & (y_test == -1), 1, 0))
    trueNegatives = np.sum(np.where((y_pred == -1) & (y_test == -1), 1, 0))
    falseNegatives = np.sum(np.where((y_pred == -1) & (y_test == 1), 1, 0))

    accuracy = (truePositives + trueNegatives) / y_test.shape[0]
    missRate = 1 - accuracy

    data = {
        '': ['email (true label)', 'spam (true label)'],
        'email (predicted)': [truePositives, falsePositives],
        'spam (predicted)': [falseNegatives, trueNegatives],
    }

    df = pd.DataFrame(data)
    print(df)
    print('Accuracy: ', accuracy)
    print('Miss-classification Rate: ', missRate)

    return 0 # STUB

def deployModel(filename, lr=0.01, iter=1000):

    # Load and preprocess the data
    X_train, y_train, X_test, y_test = loadData('spambase.data')

    # Train the model
    weights = logistic_regression(X_train, y_train, lr, iter)

    # Make predictions
    y_pred = sigmoid(np.dot(np.hstack((np.ones((X_test.shape[0], 1)), X_test)), weights))
    y_pred = np.where(y_pred > 0.5, 1, -1)

    evaluationMetrics(y_pred, y_test)



# Main function
if __name__ == "__main__":
    deployModel('spambase.data')
    
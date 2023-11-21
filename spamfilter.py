import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



feature_names = np.array([
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total"
])

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

def show_table(df, accuracy, missRate):
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')

    # Display accuracy and miss-classification rate below the table
    plt.text(0.5, -0.05, f'Accuracy: {accuracy:.2f}', ha='center', transform=ax.transAxes)
    plt.text(0.5, -0.1, f'Miss-classification Rate: {missRate:.2f}', ha='center', transform=ax.transAxes)

    plt.show()

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

    show_table(df, accuracy, missRate)
    
def show_feature_importance(weights, feature_names):
    # Exclude the bias term from the weights and flatten the array
    feature_weights = weights.flatten()

    # Combine weights with their corresponding feature names
    sorted_indices = np.argsort(feature_weights)[::-1]

    # Prepare data for the top 10 features
    top_features = feature_names[sorted_indices[:10]]
    top_weights = feature_weights[sorted_indices[:10]]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(top_features, top_weights, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Weights')
    plt.title('Top 10 Most Important Features')
    plt.xticks(rotation=45)

    # Add the weight values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')


    plt.show()


def deployModel(filename, ftNames,lr=0.01, iter=1000):

    # Load and preprocess the data
    X_train, y_train, X_test, y_test = loadData('spambase.data')

    # Train the model
    weights = logistic_regression(X_train, y_train, lr, iter)

    # Make predictions
    y_pred = sigmoid(np.dot(np.hstack((np.ones((X_test.shape[0], 1)), X_test)), weights))
    y_pred = np.where(y_pred > 0.5, 1, -1)

    evaluationMetrics(y_pred, y_test)
    show_feature_importance(weights, ftNames)

# Main function
if __name__ == "__main__":
    deployModel('spambase.data', feature_names)
    
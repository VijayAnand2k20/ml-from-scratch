# Other modules only for testing purposes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Import our model
from logistic_regression import LogisticRegression

# Get data
dataset = datasets.load_breast_cancer()

# Make some data
X, y = dataset.data, dataset.target

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# creating the instance of the model
regressor = LogisticRegression(learning_rate=0.0001, num_of_iterations=1000)

# training the model
regressor.fit(training_data=X_train, training_labels=y_train)

# getting the predictions
predictions = regressor.predict(data=X_test)

print("Logistic Regression classification accuracy:", accuracy(y_true=y_test, y_pred=predictions))
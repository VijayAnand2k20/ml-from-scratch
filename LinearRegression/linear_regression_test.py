import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Import our model
from linear_regression import LinearRegression

# Make some data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4) # n_samples is the number of samples and n_features is the number of features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# regressor = LinearRegression(learning_rate=0.001)
regressor = LinearRegression(learning_rate=0.01)

regressor.fit(X_train, y_train)
predicted_values = regressor.predict(X_test)

# To see how well our model performs
def mean_squared_error(actual_value, predicted_value):
    return np.mean((actual_value - predicted_value)**2)

mse_value = mean_squared_error(y_test, predicted_values)
print(mse_value)

# Plotting
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m1 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['#ff0000', '#00ff00', '#0000ff'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_train.shape) # This is an ndarray of shape (120,4) ==> 120 samples with 4 features each - Sepal Length, Sepal Width, Petal, Length, Petal, Width
# print(X_train[0])

# print(y_train.shape) # This is a 1-D row vector i.e., Shape: (120,) ==> We got 120 labels for 120 samples
# print(y_train) # The 3 classes are encoded with 0, 1, 2 - setosa, versicolor, virginica

# print(X[:, 0]) # This is the array of values of `Sepal Length` for 120 samples.
# print(X[:, 1]) # This is the array of values of `Sepal Width` for 120 samples.

# plt.figure()
# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=cmap, edgecolors='k', s=20) # c=y is what that applies the allotted color from cmap to corresponding features with their y values. For ex, if a feature point belongs to `setosa`, it is plotted with red.
# plt.show()

# from collections import Counter
# a = [1, 1, 1, 2, 2, 2, 3, 4, 5]

# most_common = Counter(a).most_common(1)
# print(most_common) # [(1,3)]
# print(most_common[0]) # (1,3)
# print(most_common[0][0]) # 1


print(X_train[5]) # prints the 6th training data features list
print(y_train[5]) # prints the label of the 6th training data
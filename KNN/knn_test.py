import numpy as np

# sklearn for loading datasets only
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Import our own model
from knn import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

classifier = KNN(k=3) # 100% accuracy
# classifier = KNN(k=5) # 96.67% accuracy
classifier.fit(training_data=X_train, training_labels=y_train)

# Testing
# predictions = classifier.predict(X_test)

# print(len(y_test))
# print(np.sum(predictions == y_test))

# acc = np.sum(predictions == y_test) / len(y_test)
# print(f"Accuracy: {acc}")

# Manual Input testing
FLOWERS = ['setosa', 'versicolor', 'virginica']

# [4.8, 3.4, 1.9, 0.2] ==> setosa
# [6.7, 3.0, 5.0, 1.7] ==> versicolor

def return_labels(flower_id):
    '''
    Args:
        flower_id: ID of the flower (0 or 1 or 2)

    Returns:
        Returns the name of the flower corresponding to the id
    '''
    return FLOWERS[flower_id]

if __name__ == '__main__':
    sepal_length = float(input("Enter the sepal length: "))
    sepal_width = float(input("Enter the sepal width: "))
    petal_length = float(input("Enter the petal length: "))
    petal_width = float(input("Enter the petal width: "))
    inputs = [
        [sepal_length, sepal_width, petal_length, petal_width]
    ]
    predictions = classifier.predict(data=inputs)
    # print(predictions[0]) ==> Gives the flower_id
    predictions = map(return_labels, predictions)
    print(f'The predicted flower is: {list(predictions)[0]}')
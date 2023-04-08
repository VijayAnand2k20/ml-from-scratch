import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    '''
    This function computes the euclidean distance between two vectors.
    It is used in the KNN algorithm to determine the distance between two points
    and assign them to the nearest cluster.
    '''
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    '''
    This is a class that implements a K-Nearest Neighbors algorithm.
    The KNN class has two methods: fit and predict. The fit method trains the classifier
    on the input data and labels, while the predict method uses the classifier to predict
    the labels of a new dataset. The KNN class also has another method, _predict, which is
    a helper method to predict the label of a single data point. The predict method calls
    this method for each data point in the dataset.

    Available Functions
    -------------------
    fit(training_data, training_labels)
        setting the values of training data and labels
    predict(data)
        Returns an array of predicted labels for the data provided
    
    '''

    def __init__(self, k=3):
        '''
        Initializes the value of k
        
        Args:
            k: integer representing the number of neighbours you want to group
        '''
        self.k = k

    def fit(self, training_data, training_labels):
        '''
        It Save the values of training data and labels

        Args:
            training_data: ndarray of the training data
            training_labels: ndarray of the labels corresponding to the training data
        
        Returns:
            It doesn't return any value
        '''
        self.X_train = training_data
        self.y_train = training_labels

    def predict(self, data):
        '''
        It predicts the label of the given data
        
        Args:
            data: ndarray of data to be predicted

        Returns:
            It returns a list of predictions for the list of data
        '''
        predicted_labels = [self._predict(datum) for datum in data]
        return np.array(predicted_labels)
    
    def _predict(self, datum):
        '''
        This is a helper method of `predict` method. It returns the label of the datum at the point where it is called

        Args:
            datum: datapoints of a datum at a point
        
        Returns:
            It returns the predicted label of the particular datum
        '''

        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get k-nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
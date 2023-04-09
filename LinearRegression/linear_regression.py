import numpy as np

class LinearRegression:
    """
    This class is used to implement the Linear Regression algorithm

    Attributes:
        learning_rate (float): The learning rate of the model
        num_of_iterations (int): The number of iterations the model will run for
        weights (numpy.ndarray): The weights of the model
        bias (float): The bias of the model

    Methods:
        fit(training_data, training_labels): This function is used to train the model on the training data and training labels
        predict(testing_data): This function is used to predict the labels for the testing data
    """

    def __init__(self, learning_rate=0.001, num_of_iterations=1000):
        '''
        This is a constructor for the LinearRegression class which is used to initialize the values of the class variables learning_rate and num_of_iterations

        Parameters:
            learning_rate (float): The learning rate of the model
            num_of_iterations (int): The number of iterations the model will run for

        Returns:
            None
        '''
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.weights = None # weights is nothing but the vaue of m in y = mx + c
        self.bias = None # bias is nothing but the vaue of c in y = mx + c

    def fit(self, training_data, training_labels):
        '''
        This function is used to train the model on the training data and training labels

        Parameters:
            training_data (numpy.ndarray): The training data
            training_labels (numpy.ndarray): The training labels

        Returns:
            None
        '''
        # init parameters
        num_of_samples, num_of_features = training_data.shape
        self.weights = np.zeros(num_of_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.num_of_iterations):
            y_predicted = np.dot(training_data, self.weights) + self.bias

            dw = (2/num_of_samples) * np.dot(training_data.T, (y_predicted - training_labels))
            db = (2/num_of_samples) * np.sum(y_predicted - training_labels)
            
            # dw = (2/num_of_samples) * np.dot(training_data.T, (y_predicted - training_labels)) is the partial derivative of the cost function with respect to the weights (m) and bias (c) respectively.

            # db = (2/num_of_samples) * np.sum(y_predicted - training_labels) is the partial derivative of the cost function with respect to the weights (m) and bias (c) respectively.

            # The cost function is the mean squared error function which is given by the formula (1/n)*sum(y_predicted - y_actual)^2

            # Update the weights and bias i.e. m(weight) and c(bias) respectively.
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

    def predict(self, testing_data):
        """
        This function is used to predict the labels for the testing data

        Parameters:
            testing_data (numpy.ndarray): The testing data

        Returns:
            numpy.ndarray: The predicted labels
        """
        predictions = np.dot(testing_data, self.weights) + self.bias
        return predictions
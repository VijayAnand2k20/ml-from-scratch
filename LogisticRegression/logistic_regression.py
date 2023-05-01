import numpy as np

class LogisticRegression:
    # Write documentations
    """
    This class is used to implement the Logistic Regression algorithm

    Attributes:
        learning_rate (float): The learning rate of the model
        num_of_iterations (int): The number of iterations the model will run for
        weights (numpy.ndarray): The weights of the model
        bias (float): The bias of the model
        
    Methods:
        fit(training_data, training_labels): This function is used to train the model on the training data and training labels
        predict(testing_data): This function is used to predict the labels for the testing data
    """

    def __init__(self, learning_rate=0, num_of_iterations=1000):
        """
        This is a constructor for the LogisticRegression class which is used to initialize the values of the class variables learning_rate and num_of_iterations

        Parameters:
            learning_rate (float): The learning rate of the model
            num_of_iterations (int): The number of iterations the model will run for

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, training_data, training_labels):
        """
        This function is used to train the model on the training data and training labels

        Parameters:
            training_data (numpy.ndarray): The training data
            training_labels (numpy.ndarray): The training labels

        Returns:
            None
        """

        # init parameters
        num_of_samples, num_of_features = training_data.shape
        self.weights = np.zeros(num_of_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_of_iterations):
            linear_model = np.dot(training_data, self.weights) + self.bias
            predicted_labels = self._sigmoid(linear_model)

            dw = (1 / num_of_samples) * np.dot(training_data.T, (predicted_labels - training_labels))
            db = (1 / num_of_samples) * np.sum(predicted_labels - training_labels)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, data):
        """
        This function is used to predict the labels for the testing data

        Parameters:
            data (numpy.ndarray): The testing data

        Returns:
            predicted_labels_classes (list): The predicted labels for the testing data
        """
        linear_model = np.dot(data, self.weights) + self.bias
        predicted_labels = self._sigmoid(linear_model)
        predicted_labels_classes = [1 if label_probability>=0.5 else 0 for label_probability in predicted_labels]
        return predicted_labels_classes

    def _sigmoid(self, x):
        """
        This function is used to calculate the sigmoid of a number

        Parameters:
            x (float): The number whose sigmoid is to be calculated

        Returns:
            1 / (1+np.exp(-x)) (float): The sigmoid of the number
        """
        return 1 / (1+np.exp(-x))
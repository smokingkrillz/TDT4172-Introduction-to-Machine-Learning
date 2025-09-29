import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
class LogisticRegression:
    """
    A simple implementation of Logistic Regression
    using Gradient Descent optimization
    """

    def __init__(self, learning_Rate=0.001, epochs=1000):
        self.learning_Rate = learning_Rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.accuracies = []
        self.losses = []
        self.y_pred_prob = None


    def sigmoid(self, x):
        """
        Sigmoid activation function

        Args:
            x (array<m>): a vector of floats

        Returns:
            A vector of floats after applying sigmoid
        """
        return 1 / (1 + np.exp(-x))

    def _loss_function(self, y, y_pred):
        """Loss function for Logistic Regression, cross-entropy loss
        Args:
            y (array<m>): true labels
            y_pred (array<m>): predicted probabilities
        Returns:
            A float representing the loss

        """
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss 
    
    
    
    def _compute_gradients(self, X, y, y_pred):
        """
        Computes the gradients of the loss function
        with respect to weights and bias
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of binary labels (0 or 1)
            y_pred (array<m>): a vector of predicted probabilities
        Returns:
            A tuple of two elements:
            - dW (array<n>): partial derivative of loss with respect to the
              weights
            - db (float): partial derivative of loss with respect to the bias
        """
        n = X.shape[0]
        error = y_pred - y
        # Correct gradient (dJ/dw = X^T (y_pred - y) / n, dJ/db = sum(y_pred - y)/n)
        dW = X.T.dot(error) / n
        db = np.sum(error) / n
        return dW, db

    def _update_parameters(self, dW, db):
        """
        Updates weights and bias using the computed gradients
        
        Args:
            dW (array<n>): gradient of the loss with respect to weights
            db (float): gradient of the loss with respect to bias
        
        """
        self.weights -= self.learning_Rate * dW
        self.bias -= self.learning_Rate * db

    def accuracy(self, y_true, y_pred):
        """
        Computes the accuracy of predictions

        Args:
            y_true (array<m>): true binary labels (0 or 1)
            y_pred (array<m>): predicted binary labels (0 or 1)

        Returns:
            A float representing the accuracy
        """

        pred_to_class = (y_pred >= 0.5).astype(int)
        correct_predictions = np.sum(y_true == pred_to_class)
        accuracy = correct_predictions / len(y_true)
        return accuracy
    
    def fit(self, X, y):
        """
        Trains the Logistic Regression model using Gradient Descent

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of binary labels (0 or 1)
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            gradient_w, gradient_b = self._compute_gradients(X, y, y_pred)

            self._update_parameters(gradient_w, gradient_b)
            loss = self._loss_function(y, y_pred)

            self.losses.append(loss)
            accuracy = self.accuracy(y, y_pred)
            self.accuracies.append(accuracy)
            
    
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                print(f"Accuracy: {accuracy}")

    def predict(self, X):
        """
        Predicts binary labels for input data X

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A vector of binary labels (0 or 1)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1) 
         
        linear_model = np.dot(X, self.weights) + self.bias
        
   
        self.y_pred_prob = self.sigmoid(linear_model)
  
        pred_to_class = (self.y_pred_prob >= 0.5).astype(int)
        
        return pred_to_class


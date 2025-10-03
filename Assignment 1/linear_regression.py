import numpy as np


class LinearRegression:
    def __init__(self, learning_Rate=0.001, epochs=1000):
        self.learning_Rate = learning_Rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

        pass

    def _loss_function(self, y, y_pred):
        """
        Mean squared error loss function

        Args:
            y (array<m>): a vector of floats
            y_pred (array<m>): a vector of floats

        Returns:
            A float representing the loss
        """
        return (1/2)*np.mean((y - y_pred) ** 2)

    def compute_gradients(self, X, y, y_pred):
        """
        Computes the gradients of the loss function
        with respect to weights and bias

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#total data) and n columns (#features)
            y (array<m>): a vector of floats
            y_pred (array<m>): a vector of floats

        Returns:
            A tuple of two elements:
            - dW (array<n>): partial derivative of loss with respect to the weights
            - db (float): partial derivative of loss with respect to the bias
        """
        m = X.shape[0]
        
        
        error = y - y_pred  # actual - predicted
        
        dW = -(1 / m) * np.dot(X.T, error) 
        db = -(1 / m) * np.sum(error) 

        return dW, db

    def update_parameters(self, dW, db):
        """
        Updates weights and bias using gradients

        Args:
            dW (array<n>): gradients w.r.t. weights
            db (float): gradient w.r.t. bias
        """
        self.weights -= self.learning_Rate * dW
        self.bias -= self.learning_Rate * db

    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        

        if X.ndim == 1:
            X = X.reshape(-1, 1)
            

        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input data contains infinite values")
 
        self.weights = np.zeros(X.shape[1]) 
        self.bias = 0.0

        print(f"Initial weights: {self.weights}")
        print(f"Initial bias: {self.bias}")
        print(f"Learning rate: {self.learning_Rate}")

        for epoch in range(self.epochs):
            # y' = (ax + b)
            y_pred = self.predict(X)

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)

            self.update_parameters(grad_w, grad_b)
            loss = self._loss_function(y, y_pred)
            self.losses.append(loss)
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Weight = {self.weights[0]:.6f},Bias = {self.bias:.6f}")

        print(f"Final weights: {self.weights}")
        print(f"Final bias: {self.bias}")
        print(f"Final loss: {self.losses[-1]:.1f}")

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats
        """

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias

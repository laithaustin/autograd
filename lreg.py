# logistic regression from scratch
import numpy as np
import matplotlib.pyplot as plt
from engine import tensor
import sklearn.datasets
import math

class LogisticRegressor:
    def __init__(self, lr=0.001, epochs=100, regularization=None):
        self.weights = None
        self.bias = None
        self.loss = []
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, x):
        # verify that x is a tensor
        x = x if isinstance(x, tensor) else tensor(x)
        # create ones array to match shape of x
        ones = tensor(np.ones(x.data.shape))
        # compute sigmoid
        return ones / (ones + (-x).exp())
        
        
    def fit(self, x, y):
        # initialize weights and bias
        weights = np.random.randn(x.shape[1], 1) * 0.01
        weights = tensor(weights)
        # make bias the same shape as x
        bias = tensor(np.zeros(x.shape[0]).reshape(-1, 1))
        # convert x and y to tensors
        x = tensor(x)
        # reshape y to be a column vector
        y = y.reshape(-1, 1)
        y = tensor(y)

        # train model
        for i in range(self.epochs):
            z = x.dot(weights) + bias
            y_hat = self.sigmoid(z)

            # define ones array to match shape of y_hat
            ones = tensor(np.ones(y_hat.data.shape))
            
            # compute log loss 
            loss = -((y * y_hat.log()) + ((ones - y) * (ones - y_hat).log()))
            loss.backward()
            self.loss.append(loss.data.mean())

            #update weights
            weights.data -= weights.grad * self.lr
            #take the means across the rows of bias
            bias.data -= bias.grad * self.lr
            
            # zero out gradients
            weights.grad *= 0
            bias.grad *= 0

            print(f"Epoch {i}: Training loss = {loss.data.mean()}") 

        self.weights = weights
        self.bias = bias

    def predict(self, x):
        x = tensor(x)
        z = x.dot(self.weights) + self.bias
        y_hat = self.sigmoid(z)
        return y_hat.data
    
    def plot(self):
        plt.plot(self.loss)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("logreg_loss.png")

# load data
x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
x = (x - x.mean(axis=0)) / x.std(axis=0)

# train model
model = LogisticRegressor(lr=0.001, epochs=100)
model.fit(x, y)

# plot loss
model.plot()

        






        




    

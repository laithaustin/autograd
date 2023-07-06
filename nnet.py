import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

from engine import tensor

# Create a neural network model from scratch
class nnet:
    def __init__(self, layers=[]):
        self.layers = layers
        self.loss = []
        
    def add(self, layer):
        self.layers.append(layer)

    # define layer class
    class layer:
        def __init__(self, input_size, output_size, activation=None):
            self.weights = tensor(np.random.randn(input_size, output_size) * 0.01, requiresGrad=True)
            self.bias = tensor(np.zeros(output_size), requiresGrad=True)
            self.activation = activation

        def forward(self, x):
            out = x.dot(self.weights) + self.bias
            return out.relu() if self.activation == 'relu' else out
        
        def update(self, lr=0.01):
            # update weights and bias
            self.weights.data -= self.weights.grad * lr
            self.bias.data -= self.bias.grad.T.mean() * lr
            # zero out gradients
            self.weights.grad *= 0
            self.bias.grad *= 0

    # forward pass for nnet
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    # update weights for nnet
    def update(self, lr=0.01):
        for layer in self.layers:
            layer.update(lr)
    
    # train nnet
    def train(self, x, y, epochs=100, lr=0.01):
        for i in range(epochs):
            out = self.forward(x)
            loss = (out - y) ** 2
            self.loss.append(loss.data.mean())
            loss.backward()
            self.update(lr)
            print(f"Epoch {i}: Training loss = {loss.data}")

    # plot nnet loss results
    def plot(self, x, y):
        plt.plot(self.loss)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        plt.savefig("loss.png")

# Create a neural network model and test
if __name__ == "__main__":
    # load in sample data from sklearn
    x, y = sklearn.datasets.make_moons(200, noise=0.20)
    x = tensor(x, requiresGrad=False)
    # make y an array of arrays
    y = np.array([[i] for i in y])
    y = tensor(y, requiresGrad=False)

    # create a 1 layer neural network model
    model = nnet()
    model.add(nnet.layer(2, 3, 'relu'))
    model.add(nnet.layer(3, 1))

    # train neural network model
    model.train(x, y, epochs=100, lr=0.01)

    # plot neural network model loss
    model.plot(x, y)

        
    


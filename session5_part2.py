#usage of nnfs (neural nexus from scratch)

import numpy as np
import nnfs

nnfs.init() # ⁡⁢⁣⁣what is it doing?⁡
#1. ⁡⁣⁣⁢Setting the random seed⁡
#2. ⁡⁣⁣⁢setting a default datatype for numpy (just for getting same values as the tutor)

# ⁡⁣⁢⁣#creating some random data⁡
# def create_data(points, classes):
#     X = np.zeros((points*classes,2)) # data matrix (each row = single example)
#     y = np.zeros(points*classes, dtype='uint8') # class labels
#     for class_number in range(classes):
#         ix = range(points*class_number,points*(class_number+1))
#         r = np.linspace(0.0,1,points) # radius
#         t = np.linspace(class_number*4,(class_number+1)*4,points) + np.random.randn(points)*0.2 # theta
#         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
#         y[ix] = class_number
#     return X,y


#⁡⁣⁢⁡⁣⁢⁣now, since we are using nnfs⁡⁣⁢⁣ we will import the spiral data directly⁡

import matplotlib.pyplot as plt

from nnfs.datasets import spiral_data



X, y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) +self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

layers1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layers1.forward(X)
#print(layers1.output)
activation1.forward(layers1.output)

print(activation1.output)

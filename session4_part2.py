import numpy as np

np.random.seed(0) 

X = [[1, 2, 3, 2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]] 


#How do we initialize a layer in neural networks

# two ways:
# the first is you've got a trained model that you saved and you actually wanna load in that model , so what actually happens when you save the model is you're just saving the weights and biases 
# In this case, we're gonna initialize weights and biases
# generally in nn, we want smaller values
# ranges between negative 1 and positive 1
# for we're gonna use random in numpy

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))    #additional info: numpy.zeros(shape, dtype=float, order='C'), The 2nd parameter should be data type and not a number
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) +self.biases

layer1 = Layer_Dense(4,5) # We specify here, the size of the inputs and how many neurons we wanna have # so the size of the inputs here, we know that is 4 (no. of features in each sample)
# And then for the number of neurons, is anything you want (any no. basically) let's say 5
# Similarly let's specify the layer 2
# The only requirement for layer 2 is that the output from layer 1 is gonna be the input to layer 2
# therefore, the no. of inputs's shape have to 5
# but again, the output can be of any shape that you want, let's say 2

layer2 = Layer_Dense(5,2)

layer1.forward(X)

print(layer1.output)

# layer 1 output becomes the input of layer 2 so, 


layer2.forward(layer1.output)
print(layer2.output)
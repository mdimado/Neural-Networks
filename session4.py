import numpy as np

inputs = [[1, 2, 3, 2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]



layer_outputs = np.dot(inputs,np.array(weights).T) + biases

#print(f" {layer_outputs}")  #hidden layer

# error before transposing the weights matrix:
#ValueError: shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)

#output after transposing the weights matrix:
#s [[ 4.8    1.21   2.385]
 #[ 8.9   -1.81   0.2  ]
 #[ 1.41   1.051  0.026]]



 #let's add another layer

weights2 = [[0.1, -0.14, 0.5],[-0.5, 0.12, -0.33],[-0.44, 0.73, -0.13]]
biases2 = [-1,2,-0.5]

layer1_outputs = layer_outputs

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) +biases2
print(layer2_outputs)
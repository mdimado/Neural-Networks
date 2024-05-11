
#softman activatiojn for batch of neurons

import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

# print(np.sum(layer_outputs)) #gives sum of all the elements

# print(np.sum(layer_outputs, axis=0)) #gives an array of sums of each individual row

# print(np.sum(layer_outputs, axis=1)) #gives an array of sums of each individual column

# print(np.sum(layer_outputs, axis=1, keepdims=True)) #gives the sum but with the same dimensions

#so, norm values:

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)


print(norm_values)
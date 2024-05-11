import math
import numpy as np

layer_outputs = [4.8,1.21,2.385]

#let's code the raw python implementation for exponentiation

#first, we need a defenition for e

#E = 2.71828182846

E = math.e

#once we have that number , we wanna exp all the output

exp_values = np.exp(layer_outputs)

# for output in layer_outputs:
#     exp_values.append(E**output)

# print(exp_values)



#let's go ahead and code the normalization
# normalization is the single values divided by the total values

norm_base = sum(exp_values)

norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print(sum(norm_values))
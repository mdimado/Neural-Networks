
# What if we want to model three neurons with four inputs each XD
#this means, we're gonna have four inputs
#since there's three neurons there's going to be three unique weight sets and each weight set is going to have four unique weights
#also, we're going to need 3 unique or separate biases

inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
inputs[0]*weights2[0] + inputs [1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
inputs[0]*weights3[0] + inputs [1]*weights3[1] + inputs[2] *weights3[2] + inputs[3]*weights3[3] + bias3]

print (output)
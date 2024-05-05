"""
versions used:
Python: 3.7.7
Numpy: 1.18.2
Matplotlib: 3.2.1
"""

#We're creating a neuron yay!! (let's just say this neuron is getting input from 3 neurons)

inputs = [1.2,5.1,2.1] #outputs from 3 neurons
weights = [3.1,2.1,8.7] #every input has unique weight associated with it
bias = 3 #every unique neuron has a unique bias

#So, now, the 1st step for neuron is to addup all the inputs times the weights plus bias

# inputs[0]+weights[0] + inputs[1]+weights[1] + inputs[2]+weights[2] + ... + inputs[n]+weights[n]

output = 0
for i in range(len(inputs)):
    output = output + (inputs[i]*weights[i])

output = output +bias
print(output)
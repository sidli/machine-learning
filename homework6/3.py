import numpy as np
import random

def generate_input(size):
    inputs = []
    outputs = []
    for i in range(size):
        xs = []
        for j in range(10):
            xs.append(random.randint(0,1))
        if np.sum(np.array(xs)) % 4 == 0:
            outputs.append(1)
        else: 
            outputs.append(0)
        inputs.append(xs)
    return np.array(inputs),np.array(outputs)

class NeuralNet(object):
    def __init__(self): 
        self.l2_weights = np.reshape(np.random.rand(3 * 10), (10,3))
        self.l3_weights = np.reshape(np.random.rand(3),(3,))

    # The Sigmoid function 
    def sigmoid(self, x): 
        return 1.0 / (1.0 + np.exp(-x))

    # The derivative of the Sigmoid function. 
    def sigmoid_derivative(self, x): 
        return x * (1 - x) 

    # Train the neural network and adjust the weights each time. 
    def train(self, inputs, outputs, size):
        training_iterations = size * 10
        for iteration in range(training_iterations):
            # Pass the training set through the network.
            index = random.randint(0, size-1)
            z2, a2, a3 = self.learn(inputs[index])
            # Calculate the error
            factor3 = (a3 - outputs[index]) * a2
            # Adjust the weights by a factor
            self.l3_weights -= 0.01 * factor3
            factor2 = np.reshape(self.l3_weights, (3,-1)) * (np.reshape(self.sigmoid_derivative(z2) * (a3 - outputs[index]), (3,1)) * inputs[index])
            self.l2_weights -= 0.01 * factor2.T
            self.l3_weights[self.l3_weights < -1] = -1
            self.l2_weights[self.l2_weights < -1] = -1
            self.l3_weights[self.l3_weights > 1] = 1
            self.l2_weights[self.l2_weights > 1] = 1

    # The neural network thinks.
    def learn(self, inputs):
        l2_z = np.dot(inputs, self.l2_weights)
        l2_a = self.sigmoid(l2_z)
        return l2_z ,l2_a, self.sigmoid(np.dot(l2_a, self.l3_weights))

if __name__ == "__main__": 
    #Initialize 
    neural_network = NeuralNet()
    # The training set.
    size = 100
    inputs, outputs = generate_input(size)
    print(inputs)
    print(outputs)
    # Train the neural network 
    neural_network.train(inputs, np.reshape(outputs,(-1,1)), size)
    # Test the neural network with a test example. 
    print(neural_network.learn(np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1])))

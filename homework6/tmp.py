from numpy import *
  
class NeuralNet(object): 
    def __init__(self): 
        # Generate random numbers 
        random.seed(1) 
  
        # Assign random weights to a 3 x 1 matrix, 
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
  
    # The Sigmoid function 
    def __sigmoid(self, x): 
        return 1 / (1 + exp(-x)) 
  
    # The derivative of the Sigmoid function. 
    # This is the gradient of the Sigmoid curve. 
    def __sigmoid_derivative(self, x): 
        return x * (1 - x) 
  
    # Train the neural network and adjust the weights each time. 
    def train(self, inputs, outputs, training_iterations): 
        for iteration in range(training_iterations): 
  
            # Pass the training set through the network. 
            output = self.learn(inputs) 
            print("output:",output)
  
            # Calculate the error 
            error = outputs - output 
            print("error:",error)
  
            # Adjust the weights by a factor
            print(inputs.T)
            print(error * self.__sigmoid_derivative(output))
            print(error)
            print(self.__sigmoid_derivative(output))
            factor = dot(inputs.T, error * self.__sigmoid_derivative(output)) 
            print("factor:",factor)
            self.synaptic_weights += factor 
            exit(0)
  
    # The neural network thinks. 
    def learn(self, inputs): 
        return self.__sigmoid(dot(inputs, self.synaptic_weights)) 
  
if __name__ == "__main__": 
  
    #Initialize 
    neural_network = NeuralNet() 
  
    # The training set. 
    inputs = array([[0, 1, 1], [1, 0, 0], [1, 0, 1], [1,1,1]]) 
    outputs = array([[1, 0, 1, 1]]).T 
  
    # Train the neural network 
    neural_network.train(inputs, outputs, 10000) 
  
    # Test the neural network with a test example. 
    print(neural_network.learn(array([1, 0, 1])))

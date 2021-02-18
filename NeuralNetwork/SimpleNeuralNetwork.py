# trying to implement a simple neural network
import numpy as np


class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output

    def train(self, training_input, training_output, training_times):
        for times in range(training_times):
            output = self.think(training_input)
            error = training_output - output
            adjustments = np.dot(training_input.T, self.sigmoid_derivative(output) * error)
            self.weights += adjustments


neural_network = NeuralNetwork()
print("Randomly initialize weights for neural network")
print(neural_network.weights)
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 0, 0]])
training_outputs = np.array([[0, 1, 0, 1]]).T
neural_network.train(training_inputs, training_outputs, 100000)
print("weights after train")
print(neural_network.weights)
user_input_one = str(input("Input one: "))
user_input_two = str(input("Input two: "))
user_input_three = str(input("Input three: "))
print("New input is: ", user_input_one, user_input_two, user_input_three)
print("Output: ", neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))

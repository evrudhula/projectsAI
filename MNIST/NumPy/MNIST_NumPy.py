'''
    Neural Network in NumPy to classify the MNIST Dataset.
    Accuracy: 89.14% after 20 iterations
'''
# Using TensorFloe Initially to sort out tthe initial data. Actual NN uses only numpy. Promise.
import tensorflow as tf
import numpy as np
from tensorflow import keras
# Getting the dataset from keras libraries
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Flattening the 28x28 image data into a list of 784 data items. And then normalizing them for each to have a value between 0 and 1
RESHAPED = 784
x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = (x_train/255).astype('float32')
x_test = (x_test/255).astype('float32')

# One-Hot Encoding of the results dataset
# As there are 10 classes, one for each digit
NB_CLASSES = 10
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)


class NeuralNet():
    def __init__(self, sizes, epochs=20, l_rate = 0.025, load = False):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        if load:
            self.params = self.loadparams()
        else:
            self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        # Using the sigmoid activation function
        if derivative:
            return self.sigmoid(x)*(1-self.sigmoid(x))
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # This is logistic regression. Ofcourse we use softmax.
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps/np.sum(exps, axis=0)

    def initialization(self):
        # Number of input neurons, basically size of one training data sample
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]  # Number of neurons in hidden 1 layer
        hidden_2 = self.sizes[2]  # Number of neurons in hidden 2 layer
        output_layer = self.sizes[3]  # Number of output classes.

        params = {
            'W1': np.random.randn(hidden_1, input_layer),
            'W2': np.random.randn(hidden_2, hidden_1),
            'W3': np.random.randn(output_layer, hidden_2)
        }
        return params

    def feedforward(self, x_train):
        params = self.params
        params['A0'] = x_train  # input layer data

        # now, we enter into hidden layer 1
        params['Z1'] = np.dot(params['W1'], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])
        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backpropogation(self, y_train, output):
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        # Dividing by output.shape[0] (the number of elements in output), which is finding mean change per test sample, basically, dividing by 
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def gradien_descent(self, changes_to_w):

        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_test, y_test):
        predictions = []

        for x, y in zip(x_test, y_test):
            output = self.feedforward(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val, load=False):
        if load:
            for iteration in range(self.epochs):
                accuracy = self.compute_accuracy(x_test, y_test)
                print('Epoch ', iteration + 1, ' Accuracy: ', accuracy*100)
        else:
            for iteration in range(self.epochs):
                for x, y in zip(x_train, y_train):
                    output = self.feedforward(x)
                    changes_to_w = self.backpropogation(y, output)
                    self.update_parameters(changes_to_w)

                accuracy = self.compute_accuracy(x_val, y_val)
                print('Epoch ', iteration + 1, ' Accuracy: ', accuracy*100)
            self.saveparams()

    def saveparams(self):
        np.savetxt('W1.csv', self.params['W1'], delimiter=',')
        np.savetxt('W2.csv', self.params['W2'], delimiter=',')
        np.savetxt('W3.csv', self.params['W3'], delimiter=',')

    def loadparams(self):
        params = {
        }
        params['W1'] = np.loadtxt('W1.csv', delimiter=',')
        params['W2'] = np.loadtxt('W2.csv', delimiter=',')
        params['W3'] = np.loadtxt('W3.csv', delimiter=',')
        return params


nn = NeuralNet(sizes=[784, 128, 64, 10])
nn.train(x_train, y_train, x_test, y_test) # For training
nn.train(x_train, y_train, x_test, y_test, load = True) # For using trained model
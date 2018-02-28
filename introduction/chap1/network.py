# implementation of a NN class as exposed in M. Nielsen tutorial:
# Neural Networks and Deep Learning
# emphasis:
# - implementation of backpropagation algorithm
# - activation function
# - cost function
# - hyper-parameters tuning

# tools
import random
import numpy as np
# reproducibility
np.random.seed(123)  # seed for weights / bias generation


# implementation of the NN as a class
class Network(object):
    # parameters
    def __init__(self, sizes):
        """
        :param sizes: (list) number of neurons in each layer of the network
        [784, N1, N2, ..., 10]. The first number is fixed (784) due to the
        vector nature of each MNIST element
        """
        # number of layers from the sizes argument
        self.number_layers = len(sizes)
        # sizes translated into an attribute
        self.sizes = sizes
        # random weights and biases allocation from the normal distribution
        # weights matrix
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # bias column vector
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]

        # include the tracking of the cost function
        # self.cost = cost

    # definition of the feed-forward step: matrix product as input for the
    # activation function
    def feedforward(self, a):
        """
        calculate activation function given weights and biases
        :param a: activation of the neuron
        :return: sigmoid activation function
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # definition of the SGD method
    def sgd(self, training_data, number_epochs, mini_batch_size, lr,
            test_data=None, random_seed=123):
        """
        implement the Stochastic Gradient Descent using mini batches
        :param training_data: (array-like) list of arrays of MNIST train elements
        :param number_epochs: (int) number of epochs to train
        :param mini_batch_size: (int) size of the mini-batch to implement SGD
        :param lr: (float) learning rate parameter
        :param test_data: (array-like) list of arrays of MNIST test elements
        :param random_seed: (int) random seed for reproducibility
        :return: accuracy (array-like) list of accuracy values for each
                 training epoch
        """
        # map the arguments to lists
        training_data = list(training_data)
        # to confirm we have 50000 examples for training
        n_train = len(training_data)
        # if you include the option for test_data, measure its size
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        # reproducibility
        random.seed(random_seed)
        # list to save accuracy values after each epoch
        accuracy = []
        # iteration in the number of epochs
        for j in range(number_epochs):
            # random shuffle of the training data
            random.shuffle(training_data)
            # define the mini-batch given the batch_size argument
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n_train, mini_batch_size)]
            # iterate over each mini batch in the sample
            for mini_batch in mini_batches:
                # update the mini batch given the learning rate parameter
                self.update_mini_batch(mini_batch, lr)
            # if you deliver test_data, display information
            if test_data:
                print("Training epoch {0}: {1} / {2}  Accuracy: {3}".format(
                      j + 1,
                      self.evaluate(test_data),
                      n_test,
                      self.evaluate(test_data)/n_test))
                accuracy.append(self.evaluate(test_data) / n_test)
            else:
                print("Epoch {0} complete".format(j))
        return accuracy

    # definition of the update step for the mini-batch
    def update_mini_batch(self, mini_batch, lr):
        """
        function to update the weights and biases during gradient descent
        :param mini_batch: (array-like) mini batch of training examples
        :param lr: (scalar) learning rate parameter
        :return: None, simply updates the mini-batch sample
        """
        # allocation for the derivatives of the parameter space
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        gradient_bias = [np.zeros(b.shape) for b in self.biases]
        # iteration over the mini-batch
        for x, y in mini_batch:
            # change in the derivative
            delta_gradient_weights, \
                delta_gradient_bias = self.backpropagation(x, y)
            gradient_weights = [nw + dnw
                                for nw, dnw in zip(gradient_weights,
                                                   delta_gradient_weights)]
            gradient_bias = [nb + dnb
                             for nb, dnb in zip(gradient_bias,
                                                delta_gradient_bias)]
        # update the weights
        self.weights = [w - (lr/len(mini_batch)) * nw
                        for w, nw in zip(self.weights, gradient_weights)]
        self.biases = [b - (lr/len(mini_batch)) * nb
                       for b, nb in zip(self.biases, gradient_bias)]

    # definition of the backpropagation step
    def backpropagation(self, x, y):
        """
        return a tuple for the gradients in the weights and the biases
        :param x:
        :param y:
        :return:
        """
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        gradient_bias = [np.zeros(b.shape) for b in self.biases]
        # feed-forward step
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        # iterate over the
        for w, b in zip(self.weights, self.biases):
            # calculating the activation
            z = np.dot(w, activation) + b
            # append the value to the layer
            zs.append(z)
            # calculating the sigmoid profile
            activation = sigmoid(z)
            # and append the value of the activation
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # update the values of the derivatives
        gradient_bias[-1] = delta
        gradient_weights[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        # for l in xrange(2, self.num_layers):
        for l in range(2, self.number_layers):
            # update activation
            z = zs[-l]
            # evaluating derivative
            sp = sigmoid_prime(z)
            # change
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            gradient_bias[-l] = delta
            gradient_weights[-l] = np.dot(delta, activations[-l - 1].transpose())
        # return the tuple with the derivatives in the parameter space
        return gradient_weights, gradient_bias

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.

        :param test_data: (array-like)
        :return:
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # definition of the derivative of the cost function
    @staticmethod
    def cost_derivative(output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y


# complementary functions
def sigmoid(z):
    """
    The sigmoid function.
    :param z: (scalar) activity of the neuron
    :return: (scalar) sigmoid activation function
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.
    :param z: (scalar) activity of the neuron
    :return: (scalar) derivative of the sigmoid function
    """
    return sigmoid(z) * (1 - sigmoid(z))

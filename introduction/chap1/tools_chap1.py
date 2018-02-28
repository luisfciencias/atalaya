# tools for the source code in chapter 1
import pickle as cpickle
import gzip
import numpy as np


# load the mnist data set
# reading the pickle file with the format from 
# http://deeplearning.net/data/mnist/mnist.pkl.gz
def load_mnist():
    """
    read the pickle file from the mnist data set and outputs the tuples of
    numpy arrays for training, testing and validation. Respectively:
    X = shape(50000, 784) , shape(10000, 784) , shape(10000, 784)
    y = shape(50000,) , shape(10000,) , shape(10000,)
    :return: (X_tr, y_tr) , (X_val, y_val) , (X_test, y_test)
    """
    # path to data file
    path2mnist = '../../data/mnist.pkl.gz'
    f = gzip.open(path2mnist, 'rb')
    # it was necessary to include the encoding flag 'latin1' for py3
    train_set, valid_set, test_set = cpickle.load(f, encoding='latin1')
    f.close()
    return train_set, valid_set, test_set


# adapt the input data to the NN
def load_data_wrapper():
    """
    reads the MNIST data set and reshapes it to include
    one-hot encoding column vectors
    :return:
    """
    # read the tuples for training, validation and testing
    tr_d, va_d, te_d = load_mnist()
    # after this:
    # tr_d = tuple of arrays: 50000 X 784 and 50000
    # va_d =    "   "    "    10000 X 784 and 10000
    # te_d =    "   "    "    10000 X 784 and 10000

    # reshape the training as column vectors
    # list of 50000 column vectors of size (784,1)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # list of 10000 column vectors with one-hot encoding
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # merging the training data into an iterable object
    training_data = zip(training_inputs, training_results)
    # similar steps for validation
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    # reshaping of the test data
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    # merge them into iterable object
    test_data = zip(test_inputs, te_d[1])
    # output the iterable sets
    return training_data, validation_data, test_data


def vectorized_result(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network.
    :param j:
    :return:
    """
    # simple one-hot enconding as column vector
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

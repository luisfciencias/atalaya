# source code for the implementation of a NN for digit recognition

# tools
import network
from tools_chap1 import load_mnist, load_data_wrapper
import matplotlib.pyplot as plt

# reading the mnist data set, each term is a tuple for X, y
train_set, valid_set, test_set = load_mnist()

X_train = train_set[0]
y_train = train_set[1]

X_valid = valid_set[0]
y_valid = valid_set[1]

X_test = test_set[0]
y_test = test_set[1]

print('The shapes of the MNIST arrays are:')
print('Training:')
print('X:', X_train.shape)
print('y:', y_train.shape)

print('Validation:')
print('X:', X_valid.shape)
print('y:', y_valid.shape)

print('Testing:')
print('X:', X_test.shape)
print('y:', y_test.shape)

# including the reshaping tool for one-hot encoding
# the output of this function are zip iterable objects
train_set, valid_set, test_set = load_data_wrapper()

# generating a network of 30 --> 10 units
net = network.Network([784, 30, 10])

# training the network given the number of epochs and the size of the batch
accuracy = net.sgd(train_set, 30, 10, 3.0, test_data=test_set)

plt.plot(accuracy, 'r-',  label=r'NN1')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('train_history.png', dpi=128)
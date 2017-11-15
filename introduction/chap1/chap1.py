# source code for the implementation of a NN for digit recognition

# tools
from tools_chap1 import load_mnist
import numpy as np

# reading the mnist data set, each term is a tuple for X, y
train_set, valid_set ,test_set = load_mnist()

X_train = train_set[0]
y_train = train_set[1]

X_valid = valid_set[0]
y_valid = valid_set[1]

X_test = test_set[0]
y_test = test_set[1]

print('The shapes of the mnist arrays are:')
print('Training:')
print('X:',X_train.shape)
print('y:',y_train.shape)

print('Validation:')
print('X:',X_valid.shape)
print('y:',y_valid.shape)

print('Testing:')
print('X:',X_test.shape)
print('y:',y_test.shape)

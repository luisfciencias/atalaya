# MNIST minimal keras example

# reproducibility
import numpy as np
random_seed = 123
np.random.seed(random_seed)

# tools
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import __version__ as version
import matplotlib.pyplot as plt

print('Keras version:', version)

# input data tensors: MNIST data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# shape of the tensor arrays
print('Shape of training arrays before pre-processing:')
print(X_train.shape, y_train.shape)

# data pre-processing:
# map to floats, unit normalisation, tensor re-shaping and map to categorical
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

X_train = X_train.reshape((60000, 784))
X_test = X_test.reshape((10000, 784))

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print('Training arrays after processing:')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)

print('Testing arrays:')
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

# model/architecture definition: simple example
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=784))
model.add(Dense(units=10, activation='softmax'))

# configuration of the training process
# -- loss function definition
# -- optimization strategy
# -- output metrics
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# training model strategy: epochs + batch size
number_epochs = 50
batch_size = 32
training = model.fit(X_train, y_train, epochs=number_epochs,
                     batch_size=batch_size, validation_data=(X_test, y_test))

# to plot the model training history
plt.subplot(211)
plt.plot(range(1, number_epochs + 1), training.history['loss'],     'b-', label='training')
plt.plot(range(1, number_epochs + 1), training.history['val_loss'], 'r-', label='testing')
plt.ylabel('Loss')

plt.subplot(212)
plt.plot(range(1, number_epochs + 1), training.history['acc'],     'b-', label='training')
plt.plot(range(1, number_epochs + 1), training.history['val_acc'], 'r-', label='testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('training_history.png', dpi=128)

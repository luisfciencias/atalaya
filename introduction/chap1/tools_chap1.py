# tools for the source code in chapter 1
import pickle as cPickle
import gzip

# load the mnist data set
# reading the pickle file with the format from 
# http://deeplearning.net/data/mnist/mnist.pkl.gz
def load_mnist():
    ''' 
    read the pickle file from the mnist data set and outputs the 
    numpy arrays for training, testing and validation. Respectively: 
    X = shape(50000, 784) , shape(10000, 784) , shape(10000, 784)
    y = shape(50000,) , shape(10000,) , shape(10000,)
    '''
    path2mnist = '../../data/mnist.pkl.gz'
    f = gzip.open(path2mnist, 'rb')
    # it was necessary to include the encoding flag 'latin1' for py3
    train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    f.close()
    return train_set, valid_set, test_set


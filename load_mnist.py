# Author: Thang Vu
# Date: 25/Nove/2017
# Description: Load datasets

import gzip
from six.moves import cPickle as pickle
import os

def load_mnist_datasets(path='data/mnist.pkl.gz'):
    if not os.path.exists(path):
        raise Exception('Cannot find %s' %path)
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = pickle.load(f)
        return train_set, val_set, test_set

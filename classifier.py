# Author: Thang Vu
# Date: 25/Nov/2017
# Description: classifier for train and test

from load_mnist import load_mnist_datasets
from layers import *
from cnn import CNN
from sgd import SGD

# 
def main():
    # load datasets
    path = 'data/mnist.pkl.gz'
    train_set, val_set, test_set = load_mnist_datasets(path)
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set

    batch_size = 100
    

    cnn = CNN()
    sgd = SGD(1e-3, 1) #fake momentum

    for epoch in range(1):
        # shuffle data
        num_train = X_train.shape[0]
        num_batch = num_train//batch_size
        for batch in range(num_batch):
            batch_mask = np.random.choice(num_train, self.batch_size)
            X_batch = X_train[batch_mask]
            y_batch = y_train[batch_mask]
            
            # forward
            output = cnn.forward(X_batch)
            loss, dout = cross_entropy_loss(output, y_batch)

            # backward
            grads = sgd.step()
            cnn.update(grads)





    
    

    
    


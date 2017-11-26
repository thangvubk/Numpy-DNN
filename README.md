# Deep Neural Network implementation with Numpy

This is my implementation of Computer Vision course at KAIST. 
In this implementation, NO deep learning library is used. 

```
+-------------+    +--------+    +------+    +--------+    +--------+    +------+    +----------+    +-------+
| Input 28*28 |--->| fc 256 |-+->| Relu |--->|drop out|--->| fc 256 |-+->| ReLU |--->| Drop out |--->| fc 10 |
+-------------+    +--------+ |  +------+    +--------+    +--------+ |  +------+    +----------+    +-------+
                              |                                       |
                              +---------------------------------------+
                                         Resudual connection
```

## Setup
Dependencies:
* numpy
* six

Install dependencies: ``pip install -r requirements.txt``

## Execute
Get data (mnist.pkl.gz will be downloaded to data/)
```
cd data/
sh get_data.sh 
```
Train, validate, and test ``python main.py``

You will get approximately 98% accuracy on test data.

## Project explaination
* ``load_mnist.py``: reads data/mnist.pkl.gz to train, val, and test datasets
* ``layers.py``: declares cross entropy loss function, fully connected (fc), ReLU, and Dropout layers. A naive version for convolutional layer is also implemented but it takes quite long time for training, and this layer is not included in my network architecture. :D
* ``sgd.py``: implements Stochastic Gradient Descent with momentum, learning rate decay, and weight decay
* ``dnn.py``: defines the network architecture with fully connected, ReLU, Dropout, and residual connection
* ``main.py``: train the data, then store the best model based on validation set, finally evaluate the best model on test data.

## Contact
If you have any issues, feel free to contact me: thangvubk@kaist.ac.kr

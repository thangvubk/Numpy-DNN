# Author: Thang Vu
# Date: 26/Nov/2017
# Description: Define network architecture of 3 layers with dropout
#              residual architecture

import numpy as np
from layers import *
class DNN(object):
    """
    +-------------+    +--------+    +------+    +--------+    +--------+    +------+    +----------+    +-------+
    | Input 28*28 |--->| fc 256 |-+->| Relu |--->|drop out|--->| fc 256 |-+->| ReLU |--->| Drop out |--->| fc 10 |
    +-------------+    +--------+ |  +------+    +--------+    +--------+ |  +------+    +----------+    +-------+
                                  |                                       |
                                  +---------------------------------------+
                                             Residual connection
    """
    def __init__(self):
        """ Initialize weight and bias """
        weight_scale = 1e-3
        self.params = {}
        self.params['W1'] = weight_scale*np.random.randn(28*28, 256)
        self.params['b1'] = np.zeros(256)

        self.params['W2'] = weight_scale*np.random.rand(256, 256)
        self.params['b2'] = np.zeros(256)

        self.params['W3'] = weight_scale*np.random.rand(256, 10)
        self.params['b3'] = np.zeros(10)

        self.caches = ()
        self.mode = 'train'
        self.prob = 0.5

    def forward(self, X):

        # layer 1
        a_fc1, cache_fc1 = fc_forward(X, self.params['W1'], self.params['b1'])
        a_relu1, cache_relu1 = relu_forward(a_fc1)
        a_drop1, cache_drop1 = dropout_forward(a_relu1, self.prob, self.mode)
        
        
        #layer 2
        a_fc2, cache_fc2 = fc_forward(a_drop1, self.params['W2'], self.params['b2'])
        a_fc2 = a_fc2 + a_fc1 # residual connection
        a_relu2, cache_relu2 = relu_forward(a_fc2)
        a_drop2, cache_drop2 = dropout_forward(a_relu2, self.prob, self.mode)

        # layer 3
        a_fc3, cache_fc3 = fc_forward(a_drop2, self.params['W3'], self.params['b3'])
                
        self.caches = (cache_fc1, cache_fc2, cache_fc3,
                       cache_relu1, cache_relu2,
                       cache_drop1, cache_drop2)
        return a_fc3

    def backward(self, dout):
        grads = {}
        (cache_fc1, cache_fc2, cache_fc3,
         cache_relu1, cache_relu2,
         cache_drop1, cache_drop2) = self.caches

        # layer 3
        da_fc3, grads['W3'], grads['b3'] = fc_backward(dout, cache_fc3)

        # layer 2
        da_drop2 = dropout_backward(da_fc3, cache_drop2)
        da_relu2 = relu_backward(da_drop2, cache_relu2)
        da_fc2, grads['W2'], grads['b2'] = fc_backward(da_relu2, cache_fc2)
        
        # layer 1
        da_drop1 = dropout_backward(da_fc2, cache_drop1)
        da_relu1 = relu_backward(da_drop1, cache_relu1)
        da_relu1 = da_relu1 + da_relu2 # backward for residual connection
        da_fc1, grads['W1'], grads['b1'] = fc_backward(da_relu1, cache_fc1)

        return grads

    def train_mode(self):
        self.mode = 'train'

    def eval_mode(self):
        self.mode = 'test'






# Author: Thang Vu
# Date: 26/Nov/2017
# Description: SGD optim with learning rate decay, weight decay and momentum

from __future__ import division
class SGD(object):

    def __init__(self, lr=1e-1, lr_decay=0, weight_decay=0, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}
        self.lr_decay = lr_decay
        self.epoch = 0 # learning rate decay based on epoch
        self.weight_decay = weight_decay

    def step(self, params, grads):
        # iterate params and update
        for key in params.keys():
            self.v[key] = self.momentum*self.v.get(key, 0) - self.lr*grads[key]
            
            # momentum + weight decay
            params[key] = params[key] + self.v[key] - self.lr*self.weight_decay*params[key]

    def decay_learning_rate(self):
        self.epoch += 1
        self.lr = self.lr/(1 + self.lr_decay*self.epoch)


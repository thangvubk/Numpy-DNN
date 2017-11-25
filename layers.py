# Author: Thang Vu
# Date: 25/Nov/2017
# Description: Implement layers defined in HW1

import numpy as np

def fc_forward(x, w, b):
    """
    x shape(N, D)
    w shape(D, M)
    b shape(M)
    """
    out = x.dot(w) + b
    cache = (x, w, b)

    return out, cache

def fc_backward(dout, cache):
    x, w, b = cache

    dx = dout.dot(w.T)
    dw = np.dot(x.T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db

def relu_forward(x):
    out = np.maximum(x, 0)
    cache = x
    
    return out, cache

def relu_backward(dout, cache):
    x = cache

    dx = dout
    dx[x<=0] = 0

    return dx

def dropout_forward(x, prob, mode, seed=None):
    # use seed for gradient checking
    if seed:
        np.random.seed(seed)
    mask = None

    if mode == 'train':
        mask = np.random.rand(*x.shape)
        mask = mask < prob
        out = x/(1 - prob)
        out[mask] = 0

    elif mode == 'test':
        out = np.copy(x)

    cache = x, prob, mode, mask
    
    return out, cache

def dropout_backward(dout, cache):
    x, prob, mode, mask = cache
    
    if mode == 'train':
        dx = dout/(1 - prob)
        dx[mask] = 0
    elif mode == 'test':
        dx = dout

    return dx

def conv_forward(x, w, b, stride=1, pad=0):
    """
    - x: shape (N, C, H, W)
    - w: shape (F, C, HH, WW)
    - b: shape (F,)
    
    - out: shape (N, F, H', W')
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hout = 1 + int((H + 2*pad - HH)/stride)
    Wout = 1 + int((W + 2*pad - HH)/stride)

    #init out
    out = np.zeros((N, F, Hout, Wout))

    # zero-padding for the input. Effect on H and W only
    x_pad = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    for iN in range(N):
        for iF in range(F):
            for iH in range(Hout):
                for iW in range(Wout):
                    out[iN, iF, iH, iW] = np.sum(x_pad[iN, :, iH*stride:iH*stride+HH, iW*stride:iW*stride+WW] * w[iF]) + b[iF]
    
    cache = (x, w, b, stride, pad)
    return out, cache


def conv_backward(dout, cache):
    x, w, b, stride, pad = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hout = 1 + int((H + 2*pad - HH)/stride)
    Wout = 1 + int((W + 2*pad - HH)/stride)

    # init dw, db
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    dx = np.zeros_like(x)

    x_pad = np.lib.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    dx_pad = np.zeros_like(x_pad)

    for iN in range(N):
        for iF in range(F):
            for iH in range(Hout):
                for iW in range(Wout):
                    dw[iF] += dout[iN, iF, iH, iW] * x_pad[iN, :, iH*stride:iH*stride+HH, iW*stride:iW*stride+WW]
                    db[iF] += dout[iN, iF, iH, iW]
                    dx_pad[iN, :, iH*stride:iH*stride+HH, iW*stride:iW*stride+WW] += dout[iN, iF, iH, iW]*w[iF]

    # remove padding to archive dx
    dx = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db

def softmax_loss(x, y):
    """
    x score shape (N, C)
    y label shape (N,)
    """
    N, C = x.shape

    y_one_hot = np.eye(C)[y]
    x_shift = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x_shift)
    probs = x_exp/np.sum(x_exp, axis=1, keepdims=True)
    
    loss = -np.sum(y_one_hot*np.log(probs))/N
    dx = (probs - y_one_hot)/N

    return loss, dx


    



    

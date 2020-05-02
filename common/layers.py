#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:34:01 2019

@author: hiroakimachida
"""
import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None

        # Differential coefficient for weights and biases
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x[np.newaxis].T, dout[np.newaxis])
        self.db = dout

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax output
        self.t = None # teacher data

    def forward(self, x, t):
        self.t = t
        self.y = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
        self.loss = -np.sum(self.t * np.log(self.y + 1e-7))
        
        return self.loss

    def backward(self, dout=1):
        dx = (self.y - self.t)
        return dx
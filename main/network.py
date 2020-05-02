#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:34:01 2019

@author: hiroakimachida
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import Relu
from common.layers import Affine
from common.layers import SoftmaxWithLoss
from collections import OrderedDict


class Network:

    def __init__(self, hidden_size = 50, layer_size = 1):
        self.__input_size = 28 ** 2   # MNIST data is 28 x 28 pixel.
        self.__hidden_size = hidden_size       # Hidden layer neuron size.
        self.__layer_size = layer_size       # Hidden layer size.
        self.__output_size = 10       # Output is an one-hot array from 0 to 9.
        self.__weight_init_std = 0.01 # Standard deviation for initial weights    

        #print("neurons:"+str(self.__hidden_size))
        #print("layers:"+str(self.__layer_size))
        
        # Initialize weights and biases.
        self.params = {}
        self.params['W1'] = self.__weight_init_std * np.random.randn(self.__input_size, self.__hidden_size)
        self.params['b1'] = np.zeros(self.__hidden_size)
        for i in range(2, self.__layer_size + 1):
            self.params['W'+str(i)] = self.__weight_init_std * np.random.randn(self.__hidden_size, self.__hidden_size)
            self.params['b'+str(i)] = np.zeros(self.__hidden_size)
        self.params['Wl'] = self.__weight_init_std * np.random.randn(self.__hidden_size, self.__output_size) 
        self.params['bl'] = np.zeros(self.__output_size)

        # Generate layers.
        self.layers = OrderedDict()
        for i in range(1, self.__layer_size + 1):        
            self.layers['Affine'+str(i)] = Affine(self.params['W'+str(i)], self.params['b'+str(i)])
            self.layers['Relu'+str(i)] = Relu()
        self.layers['Affinel'] = Affine(self.params['Wl'], self.params['bl'])

        self.lastLayer = SoftmaxWithLoss()
        
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:input data, t:teacher data
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # return values
        grads = {}
        for i in range(1, self.__layer_size + 1):
            grads['W'+str(i)], grads['b'+str(i)] = self.layers['Affine'+str(i)].dW, self.layers['Affine'+str(i)].db
        grads['Wl'], grads['bl'] = self.layers['Affinel'].dW, self.layers['Affinel'].db

        return grads

    def __set_format__(self, format_):
        float_formatter = format_.format
        np.set_printoptions(formatter = {'float_kind':float_formatter})


    def repr(self):
        print("Input -> Affine(W[i]+b[i]) -> Affine(Wl+bl) -> SoftMax")
        self.__set_format__("{:.2f}")
        for key in self.params.keys():
            print(key + str(self.params[key].shape))
            #print(self.params[key])

    
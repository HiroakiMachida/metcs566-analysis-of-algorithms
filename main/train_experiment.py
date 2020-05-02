#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:34:01 2019

@author: hiroakimachida
"""
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from network import Network
#import random
#import numpy as np
#import matplotlib.pyplot as plt

def train(hidden_size = 2024, layer_size = 2):
    """
    print("")
    print("")
    print("=========================================")
    print("====   Load MNIST data")
    print("=========================================")
    """
    # Load data (used the reference implementation)
    try:
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    except:
        #print("Connect the internet.")
        sys.exit()
    # Initialize the two-layered network
    network = Network(hidden_size=hidden_size, layer_size=layer_size)
    iters_num = 50000 # Max limit of numpy is 60000
    learning_rate = 0.1
    test_acc_list = []
    i_list = []
    iter_per_epoch = 1000
    
    """
    print("Number of training iterations: " + str(iters_num))
    print("Size of training data: " + str(x_train.shape[0]))
    print("Size of test data: " + str(x_test.shape[0]))
    print("Number of iterations to check accuracy: " + str(iter_per_epoch))
    """
    
    """
    print("")
    print("")
    print("=========================================")
    print("====   Start training")
    print("=========================================")
    """
    
    iteration = tuple(range(iters_num))
    for i in iteration:
        # Calculate gradients.
        grad = network.gradient(x_train[i], t_train[i])  
        # Update parameters.
        for key in network.params:
            network.params[key] -= learning_rate * grad[key]   
        if i % iter_per_epoch == 0:
            #print("iteration: " + str(i), end = '\t')
            i_list.append(i)
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            test_acc_list.append(test_acc)
            print("Accuracy of training data: {0:.5f}".format(train_acc), end = '\t')
            #print("test data: {0:.5f}".format(test_acc))
            
    
    return max(test_acc_list)#, utils.get_size(network)

train()

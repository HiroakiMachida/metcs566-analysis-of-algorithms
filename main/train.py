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

def train(hidden_size = 50, layer_size = 1):
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
            #print("Accuracy of training data: {0:.5f}".format(train_acc), end = '\t')
            #print("test data: {0:.5f}".format(test_acc))
            
    """
    print("")
    print("")
    print("=========================================")
    print("====   Report result")
    print("=========================================")
    def show_image(img):
        for i in img:
            for j in i:
                dot = "0" if j < 125 else " "
                print(dot, end = '')
            print("")
    example_count = 5
    iteration = set(range(example_count))
    """
    """
    for i in iteration:
        print("")
        print("")
        j = random.randrange(x_test.shape[0])
        y = network.predict(x_test[j])
        predict = np.where(y == np.amax(y))
        print("Prediction:", str(predict[0]))
        z = t_test[j]
        label = np.where(z == np.amax(z))
        print("Label:", str(label[0]))
        img = x_test[j].reshape(28, 28) * 256
        show_image(img)
    """
    """
    plt.plot(i_list,test_acc_list)
    plt.title("Learning progress")
    plt.ylabel("accuracy")
    plt.xlabel("iteration")
    plt.axis(ymax = 1)
    plt.show()
    """    
    """
    print("")
    print("")
    print("=========================================")
    print("====   Network parameters")
    print("=========================================")
    network.repr()
    """
    return max(test_acc_list)#, utils.get_size(network)

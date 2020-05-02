#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:39:46 2020

@author: hiroakimachida
"""
import time
import train

import os
import psutil

process = psutil.Process(os.getpid())


combination_layer_size = [1,2,3,4,5]
combination_hidden_size = [1,2,4,8,16,32,64,128,256,512,1024,2048]

print("")
print("")
print("=========================================")
print("====   Progress")
print("=========================================")

res_accuracy = []
res_time = []
res_memory = []
for layer_size in combination_layer_size:
    sub_res_accuracy = []
    sub_res_time = []
    sub_res_memory = []
    for hidden_size in combination_hidden_size:
        start = time.perf_counter()
        accuracy = train.train(hidden_size = hidden_size, layer_size = layer_size)
        end = time.perf_counter()
        memory = process.memory_info().rss/ float(2 ** 20) #MB
        sub_res_accuracy.append(accuracy)
        sub_res_time.append(end-start)
        sub_res_memory.append(memory)
        print(f"layer_size:{layer_size} hidden_size:{hidden_size} -> accuracy: {accuracy:0.4f} time:{end - start:0.2f} seconds memory:{memory:0.0f}")
    res_accuracy.append(sub_res_accuracy)
    res_time.append(sub_res_time)
    res_memory.append(sub_res_memory)

print("")
print("")
print("=========================================")
print("====   Report")
print("=========================================")
print("x:number of neurons on each layer y:")
print("y:number of layers")
print("")

def generate_report(res_list):
    print('\t', end='')
    for i in combination_hidden_size:
        print(i, end='\t')
    print("")
    
    for i,sub_list in enumerate(res_list):
        print(str(combination_layer_size[i])+'\t', end='')
        for e in sub_list:
            print(f"{e:0.4f}", end='\t')
        print("")
    print("")
    print("")

print("Report - accuracy")
generate_report(res_accuracy)

print("Report - time")
generate_report(res_time)

print("Report - memory")
generate_report(res_memory)


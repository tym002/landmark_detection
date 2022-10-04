#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:56:19 2020

@author: tianyu
"""
import numpy as np

Y = np.load('/.../test_ground_truth.npy')
Yp = np.loadtxt(fname='/.../Prediction0122.py', delimiter=',')

l = Y.shape[0]
ld = 0.0
for i in range(0, l):
    Y1 = Y[i]
    Y1 = Y1[0]
    yl = Y1[0:3]
    ypl = Yp[i, 0:3]
    ldis = np.sqrt(np.sum(np.square((yl - ypl) * 0.5)))
    ld = ld + ldis

ald = ld / l

print("the average test distance is: ", ald, " mm")

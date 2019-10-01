#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:37:37 2019

@author: tianyu
"""
from keras.callbacks import Callback
from keras import backend as K

# This script defines the callback function on the weights of losses 

class MyCallback(Callback):
    def __init__(self, alpha, beta,gamma,delta):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if epoch <= 100:
            K.set_value(self.alpha, 16*0.8)
            K.set_value(self.beta, 4*0.2)
            K.set_value(self.gamma, 1*0)
            K.set_value(self.delta, 0.25*0)
            #K.set_value(self.gamma, K.get_value(self.gamma) - 0.0005)
            #K.set_value(self.delta, K.get_value(self.delta) + 0.0015)
        
        if 100 < epoch and epoch <= 200:
            K.set_value(self.alpha, 16*0.1)
            K.set_value(self.beta, 4*0.7)
            K.set_value(self.gamma, 1*0.2)
            K.set_value(self.delta, 0.25*0)
    
        if 200 < epoch and epoch <= 300:
            K.set_value(self.alpha, 16*0)
            K.set_value(self.beta, 4*0.1)
            K.set_value(self.gamma, 1*0.7)
            K.set_value(self.delta, 0.25*0.2)

        if 300 < epoch and epoch <= 400:
            K.set_value(self.alpha, 16*0)
            K.set_value(self.beta, 4*0)
            K.set_value(self.gamma, 1*0.2)
            K.set_value(self.delta, 0.25*0.8)

        if 400 < epoch:
            K.set_value(self.alpha, 16*0)
            K.set_value(self.beta, 4*0)
            K.set_value(self.gamma, 1*0)
            K.set_value(self.delta, 0.25*1)
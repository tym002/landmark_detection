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
    def __init__(self, alpha, beta, gamma, delta):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if epoch <= 100:
            K.set_value(self.alpha, self.alpha * 0.8)
            K.set_value(self.beta, self.beta * 0.2)
            K.set_value(self.gamma, self.gamma * 0)
            K.set_value(self.delta, self.delta * 0)

        if 100 < epoch <= 200:
            K.set_value(self.alpha, self.alpha * 0.1)
            K.set_value(self.beta, self.beta * 0.7)
            K.set_value(self.gamma, self.gamma * 0.2)
            K.set_value(self.delta, self.delta * 0)

        if 200 < epoch <= 300:
            K.set_value(self.alpha, self.alpha * 0)
            K.set_value(self.beta, self.beta * 0.1)
            K.set_value(self.gamma, self.gamma * 0.7)
            K.set_value(self.delta, self.delta * 0.2)

        if 300 < epoch <= 400:
            K.set_value(self.alpha, self.alpha * 0)
            K.set_value(self.beta, self.beta * 0)
            K.set_value(self.gamma, self.gamma * 0.2)
            K.set_value(self.delta, self.delta * 0.8)

        if 400 < epoch:
            K.set_value(self.alpha, self.alpha * 0)
            K.set_value(self.beta, self.beta * 0)
            K.set_value(self.gamma, self.gamma * 0)
            K.set_value(self.delta, self.delta * 1)

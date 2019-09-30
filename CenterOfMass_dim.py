#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:49:10 2019

@author: Tianyu
"""
from scipy import ndimage
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras import initializers as ini
from keras import regularizers

'''
This is the center of mass layer which can be trainable or fixed depending on the need. 
'''


#writing your own keras layer 
class CenterofMass(Layer):
    #centerofmass object 
    def __init__(self,size, output_dim):
        self.output_dim = output_dim
        self.size = size
        super(CenterofMass, self).__init__()

# Weight and bias term
    def build(self, input_shape):
        self.COM_weightXL = self.add_weight(name = 'COM_weightXL', shape = (self.size,1), initializer = ini.Constant(value = 0.25) , trainable = False)
        self.COM_weightXR = self.add_weight(name = 'COM_weightXR', shape = (self.size,1), initializer = ini.Constant(value = 0.25) , trainable = False)
        self.COM_weightYL = self.add_weight(name = 'COM_weightYL', shape = (self.size,1), initializer = ini.Constant(value = 0.25) , trainable = False)
        self.COM_weightYR = self.add_weight(name = 'COM_weightYR', shape = (self.size,1), initializer = ini.Constant(value = 0.25) , trainable = False)
        self.COM_weightZL = self.add_weight(name = 'COM_weightZL', shape = (self.size,1), initializer = ini.Constant(value = 0.25) , trainable = False)
        self.COM_weightZR = self.add_weight(name = 'COM_weightZR', shape = (self.size,1), initializer = ini.Constant(value = 0.25) , trainable = False)
        self.bias1 = self.add_weight(name = 'COM_bias_XL', shape = (1,1), initializer = 'zeros', trainable = False)
        self.bias2 = self.add_weight(name = 'COM_bias_XR', shape = (1,1), initializer = 'zeros', trainable = False)
        self.bias3 = self.add_weight(name = 'COM_bias_YL', shape = (1,1), initializer = 'zeros', trainable = False)
        self.bias4 = self.add_weight(name = 'COM_bias_YR', shape = (1,1), initializer = 'zeros', trainable = False)
        self.bias5 = self.add_weight(name = 'COM_bias_ZL', shape = (1,1), initializer = 'zeros', trainable = False)
        self.bias6 = self.add_weight(name = 'COM_bias_ZR', shape = (1,1), initializer = 'zeros', trainable = False)
        super(CenterofMass, self).build(input_shape)
    
    #Find the postion based on the weight and bias assinged to each heatmap center of mass. Need some explain
    def call(self, HMap): 
        '''
        calculate the center of mass. This is the untrainable version, does not use COM_weight
        '''
      List = [] 
      nx = 56
      ny = 56
      nz = 56
      i =0
      # loop_max = # of chanel from conv_layer_10 = 4
      loop_max = HMap.shape[-1]
      r1 = tf.range(0,nx, dtype = 'float32')
      r1 = K.reshape(r1, (1,nx))
      r2 = tf.range(0,ny, dtype = 'float32')
      r2 = K.reshape(r2, (1,ny))
      r3 = tf.range(0,nz, dtype = 'float32')
      r3 = K.reshape(r3, (1,nz))
      print("Loop: ", loop_max)

      while (i < loop_max):
          map1 = HMap[0,:,:,:,i]
          x = K.sum(map1[:,:,:], axis =(1,2))
          x = K.reshape(x,(nx,1))
          y = K.sum(map1[:,:,:], axis =(0,2))
          y = K.reshape(y,(ny,1))
          z = K.sum(map1[:,:,:], axis =(0,1))
          z = K.reshape(z,(nz,1))
          
          A = tf.matmul(r1,x)
          A_new = K.sum(A)
          x_new = K.sum(x)+0.00001
          CM_x = tf.divide(A_new,x_new)
          List.append(CM_x)
          
          B = tf.matmul(r2,y)
          B_new = K.sum(B)
          y_new = K.sum(y)+0.00001
          CM_y = tf.divide(B_new, y_new)
          List.append(CM_y)


          C = tf.matmul(r3,z)
          C_new = K.sum(C)
          z_new = K.sum(z)+0.00001
          CM_z = tf.divide(C_new, z_new)
          List.append(CM_z)
          
          i += 1
          
          
      List = K.stack(List)  
      List = K.reshape(List,(1,3*loop_max))
          
      
      return List
    
    def compute_output_shape(self, input_shape):

        return (input_shape[0], 1,self.output_dim)
       

  




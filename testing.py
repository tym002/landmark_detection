#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:22:30 2019

@author: sablab
"""
"""
This file compute the point location for test data. Loading pre-trained model(weight) stored 
"""


import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Dropout, AveragePooling3D, BatchNormalization,Lambda,Reshape,ZeroPadding3D    
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import losses, regularizers
from keras import backend as K
from keras import layers 
#from BBoxData_new import *
#from skimage.transform import rotate,resize, downscale_local_mean
#from sklearn.model_selection import train_test_split
import tensorflow as tf
#from CenterOfMass_dim import *
#from skimage import data
import h5py

#path of the pre-trained weight
weight_path = '/home/tm478/bif/fst_train0903b.hdf5'
#batch size
b_size = 1
#number of output channel
out_channel = 1
#number of channels for the first conv layer
ini_channel = 16
#dropout rate 
drop_rate = 0.5

#dimensions of the input image
img_rows = 448	
img_cols = 448
img_batch = 448
osize = 448

#input patch size
cx = 56
cy = 56
cz = 56

#input patch size for the last Localizer Net
lx = 56
ly = 56
lz = 56

def load_train_data():
    imgs_train = np.load('/home/tm478/bif/Save0814/Test_Mask_Coordinates448_ori_p.npy')
    return imgs_train

def Broadlayer(x):
    a = x[0,0]
    b = x[0,1]
    c = x[0,2]
    s = K.stack([0.0,0.0,0.0])
    return K.reshape(s,(1,3))

def Addlayer(x):
    p1,p2 = x
    return p1+p2

def Resize1(x):
    x1 = x[:,0::8,0::8,0::8,:]
    print("after resizing: ",x1.shape)
    return x1

def Resize2(x):
    x1 = x[:,0::4,0::4,0::4,:]
    print("after resizing: ",x1.shape)
    return x1

def Resize3(x):
    x1 = x[:,0::2,0::2,0::2,:]
    print("after resizing: ",x1.shape)
    return x1

#compute center of mass 
def COM(x):
    HMap = x
    nx = cx
    ny = cy
    nz = cz
    List = [] 
    i =0
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
        #calculating center of mass for x, which is a number  
          
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
        
    List1 = K.stack(List)  
    List1 = K.reshape(List1,(1,3*loop_max))
    return List1

def COM1(x):
    HMap = x
    nx = lx
    ny = ly
    nz = lz
    List = [] 
    i =0
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
        #calculating center of mass for x, which is a number  
          
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
        
    List1 = K.stack(List)  
    List1 = K.reshape(List1,(1,3*loop_max))
    return List1

def Padlayer(x):
    print("before cropping: ",K.shape(x))
    image_rows = K.shape(x)[1]
    image_cols = K.shape(x)[2]
    image_batch = K.shape(x)[3]
    difsizeZ = cz - image_batch
    difsizeY = cy - image_cols
    difsizeX = cx - image_rows
    Vy = tf.cond(K.greater_equal(difsizeZ,0),lambda: K.cast(difsizeZ,'int32'),lambda:0)
    Xy = tf.cond(K.greater_equal(difsizeX,0),lambda: K.cast(difsizeX,'int32'),lambda:0)
    Yy = tf.cond(K.greater_equal(difsizeY,0),lambda: K.cast(difsizeY,'int32'),lambda:0)
    return K.reshape(K.stack([Xy,Yy,Vy]),(1,3))

def Padlayer1(x):
    print("before cropping: ",K.shape(x))
    image_rows = K.shape(x)[1]
    image_cols = K.shape(x)[2]
    image_batch = K.shape(x)[3]
    difsizeZ = lz - image_batch
    difsizeY = ly - image_cols
    difsizeX = lx - image_rows
    Vy = tf.cond(K.greater_equal(difsizeZ,0),lambda: K.cast(difsizeZ,'int32'),lambda:0)
    Xy = tf.cond(K.greater_equal(difsizeX,0),lambda: K.cast(difsizeX,'int32'),lambda:0)
    Yy = tf.cond(K.greater_equal(difsizeY,0),lambda: K.cast(difsizeY,'int32'),lambda:0)
    return K.reshape(K.stack([Xy,Yy,Vy]),(1,3))

def Croplayer1(x):
    pred,img = x
    l1 =K.cast(pred[0,0,0],'int32')*2
    l2 = K.cast(pred[0,0,1],'int32')*2
    l3 =K.cast(pred[0,0,2],'int32')*2
    unpadl = img[:, K.maximum(0,l1-int(cx/2)) : K.minimum(112,l1+int(cx/2)), K.maximum(0,l2-int(cy/2)) : K.minimum(112,l2+int(cy/2)) , K.maximum(0,l3-int(cz/2)) : K.minimum(112,l3+int(cz/2)),:]
    result1 = K.cast(unpadl,'float32')
    #result = img_resize(cx,cy,cz,result1.shape[1],result1.shape[2],result1.shape[3],result1)
    return result1

def Croplayer2(x):
    pred,img = x
    l1 =K.cast(pred[0,0,0],'int32')*2
    l2 = K.cast(pred[0,0,1],'int32')*2
    l3 =K.cast(pred[0,0,2],'int32')*2
    unpadl = img[:, K.maximum(0,l1-int(cx/2)) : K.minimum(224,l1+int(cx/2)), K.maximum(0,l2-int(cy/2)) : K.minimum(224,l2+int(cy/2)) , K.maximum(0,l3-int(cz/2)) : K.minimum(224,l3+int(cz/2)),:]
    result1 = K.cast(unpadl,'float32')
    #result = img_resize(cx,cy,cz,result1.shape[1],result1.shape[2],result1.shape[3],result1)
    return result1

def Croplayer3(x):
    pred,img = x
    l1 =K.cast(pred[0,0,0],'int32')*2
    l2 = K.cast(pred[0,0,1],'int32')*2
    l3 =K.cast(pred[0,0,2],'int32')*2
    unpadl = img[:, K.maximum(0,l1-int(lx/2)) : K.minimum(448,l1+int(lx/2)), K.maximum(0,l2-int(ly/2)) : K.minimum(448,l2+int(ly/2)) , K.maximum(0,l3-int(lz/2)) : K.minimum(448,l3+int(lz/2)),:]
    result1 = K.cast(unpadl,'float32')
    #result = img_resize(cx,cy,cz,result1.shape[1],result1.shape[2],result1.shape[3],result1)
    return result1

def Computelayer(x):
    pred1,pred2 = x
    l1 = pred1[0,0,0]*2
    l2 = pred1[0,0,1]*2
    l3 = pred1[0,0,2]*2
    o1 = K.maximum(0.0,l1-cx/2)
    o2 = K.maximum(0.0,l2-cy/2)
    o3 = K.maximum(0.0,l3-cz/2)
    x1 = pred2[0,0,0]
    x2 = pred2[0,0,1]
    x3 = pred2[0,0,2]
    X = x1 + K.cast(o1,'float32')
    Y = x2 + K.cast(o2,'float32')
    Z = x3 + K.cast(o3,'float32')
    cor = [X,Y,Z]
    cor = K.stack(cor)
    cor = K.reshape(cor,(1,3))
    return cor

def Root_MSE(y_true, y_pred):
    return y_true 

# Unet Model
def PointLoc(IsBN, Drop_1, activationtype):
    inputs = Input((img_rows, img_cols,img_batch,1))
    print(type(inputs))
    reduce1 = Lambda(Resize1)(inputs)
    reduce2 = Lambda(Resize2)(inputs)
    reduce3 = Lambda(Resize3)(inputs)

    conv1 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(reduce1)
    if IsBN:
        conv1 = BatchNormalization(axis =-1, momentum =0.99)(conv1)
    #conv1 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv1)
    if Drop_1:
        conv1 = Dropout(drop_rate)(conv1)
    if IsBN:
        conv1 = BatchNormalization(axis =-1, momentum =0.99)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv1)

    conv2 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool1)
    if IsBN:
        conv2 = BatchNormalization(axis =-1, momentum =0.99)(conv2)
    #conv2 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv2)
    if Drop_1:
        conv2 = Dropout(drop_rate)(conv2)
    if IsBN:
        conv2 = BatchNormalization(axis =-1, momentum =0.99)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv2)

    conv3 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool2)
    if IsBN:
        conv3 = BatchNormalization(axis =-1, momentum =0.99)(conv3)
    #conv3 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv3)
    if Drop_1:
        conv3 = Dropout(drop_rate)(conv3)
    if IsBN:
        conv3 = BatchNormalization(axis =-1, momentum =0.99)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv3)

    conv4 = Conv3D(8*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool3)
    if IsBN:
        conv4 = BatchNormalization(axis =-1, momentum =0.99)(conv4)
    #conv4 = Conv3D(8*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv4)
    if Drop_1:
        conv4 = Dropout(drop_rate)(conv4)

    up1 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv4)
    conv4_up = concatenate([conv3,up1], axis =-1)
    conv5 = Conv3D(4*ini_channel, (3, 3, 3),activation=activationtype, padding='same', data_format = 'channels_last')(conv4_up)
    if Drop_1:
        conv5 = Dropout(drop_rate)(conv5)
    #conv6 = Conv3D(8*ini_channel, (3, 3, 3),activation=activationtype, padding='same', data_format = 'channels_last')(conv6)
   
    up2 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv5)
    conv5_up = concatenate([conv2,up2], axis =-1)
    conv6 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv5_up)
    if Drop_1:
        conv6 = Dropout(drop_rate)(conv6)
    #conv7 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv7)
    
    up3 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv6)
    conv6_up = concatenate([conv1,up3], axis =-1)
    conv7 = Conv3D(ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv6_up)
    if Drop_1:
        conv7 = Dropout(drop_rate)(conv7)
    
    conv8 = Conv3D(out_channel, (1, 1, 1), activation='relu',kernel_initializer = 'he_normal')(conv7)
    print("conv8:" ,conv8.shape)
    
    Wavg = Lambda(COM)(conv8)
    output1 = Reshape((1,3),name = "Output1")(Wavg)

    cropimg1 = Lambda(Croplayer1)([output1,reduce2])
    s1 = Lambda(Padlayer)(cropimg1)
    
    # 64*64*64
    padimg1 = Lambda(K.spatial_3d_padding,arguments={'padding':((0,s1[0,0]),(0,s1[0,1]),(0,s1[0,2]))})(cropimg1)
    conv11 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(padimg1)
    if Drop_1:
        conv11 = Dropout(drop_rate)(conv11)
    #32*32*32
    pool11 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv11)

    conv12 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool11)   
    if Drop_1:
        conv12 = Dropout(drop_rate)(conv12)
    #16*16*16
    pool12 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv12)

    conv13 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool12)    
    if Drop_1:
        conv13 = Dropout(drop_rate)(conv13)
    #8*8*8
    pool13 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv13)

    conv14 = Conv3D(8*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool13)
    if Drop_1:
        conv14 = Dropout(drop_rate)(conv14)

    up11 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv14)
    conv14_up = concatenate([conv13,up11], axis =-1)
    conv15 = Conv3D(4*ini_channel, (3, 3, 3),activation=activationtype, padding='same', data_format = 'channels_last')(conv14_up)
       
    up12 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv15)
    conv15_up = concatenate([conv12,up12], axis =-1)
    conv16 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv15_up)
    
    up13 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv16)
    conv16_up = concatenate([conv11,up13], axis =-1)
    conv17 = Conv3D(ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv16_up)    
    
    conv18 = Conv3D(out_channel, (1, 1, 1), activation='relu',kernel_initializer = 'he_normal')(conv17)
    print("conv18:" ,conv18.shape)
    
    Wavg2 = Lambda(COM)(conv18)
    List2 = Reshape((1,3))(Wavg2)
    out2 = Lambda(Computelayer)([output1,List2])
    output2 = Reshape((1,3),name = "Output2")(out2)
    cropimg2 = Lambda(Croplayer2)([output2,reduce3])
    s2 = Lambda(Padlayer)(cropimg2)
    
    # 64*64*64
    padimg2 = Lambda(K.spatial_3d_padding,arguments={'padding':((0,s2[0,0]),(0,s2[0,1]),(0,s2[0,2]))})(cropimg2)
    
    conv21 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(padimg2)
    if Drop_1:
        conv21 = Dropout(drop_rate)(conv21)
    #32*32*32
    pool21 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv21)

    conv22 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool21)   
    if Drop_1:
        conv22 = Dropout(drop_rate)(conv22)
    #16*16*16
    pool22 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv22)

    conv23 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool22)    
    if Drop_1:
        conv23 = Dropout(drop_rate)(conv23)
    #8*8*8
    pool23 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv23)

    conv24 = Conv3D(8*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool23)
    if Drop_1:
        conv24 = Dropout(drop_rate)(conv24)

    up21 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv24)
    conv24_up = concatenate([conv23,up21], axis =-1)
    conv25 = Conv3D(4*ini_channel, (3, 3, 3),activation=activationtype, padding='same', data_format = 'channels_last')(conv24_up)
       
    up22 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv25)
    conv25_up = concatenate([conv22,up22], axis =-1)
    conv26 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv25_up)
    
    up23 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv26)
    conv26_up = concatenate([conv21,up23], axis =-1)
    conv27 = Conv3D(ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv26_up)    
    
    conv28 = Conv3D(out_channel, (1, 1, 1), activation='relu',kernel_initializer = 'he_normal')(conv27)
    print("conv18:" ,conv28.shape)
    
    Wavg3 = Lambda(COM)(conv28)
    List3 = Reshape((1,3))(Wavg3)
    out3 = Lambda(Computelayer)([output2,List3])
    output3 = Reshape((1,3),name = "Output3")(out3)
    #rs = Lambda(Ranshift)(output3)
    #rs = Reshape((1,3))(rs)
    cropimg3 = Lambda(Croplayer3)([output3,inputs])
    s3 = Lambda(Padlayer1)(cropimg3)
    padimg3 = Lambda(K.spatial_3d_padding,arguments={'padding':((0,s3[0,0]),(0,s3[0,1]),(0,s3[0,2]))})(cropimg3)

    conv31 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(padimg3)
    if Drop_1:
        conv31 = Dropout(drop_rate)(conv31)
    #32*32*32
    pool31 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv31)

    conv32 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool31)   
    if Drop_1:
        conv32 = Dropout(drop_rate)(conv32)
    #16*16*16
    pool32 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv32)

    conv33 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool32)    
    if Drop_1:
        conv33 = Dropout(drop_rate)(conv33)
    #8*8*8
    pool33 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv33)

    conv34 = Conv3D(8*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool33)
    if Drop_1:
        conv34 = Dropout(drop_rate)(conv34)

    up31 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv34)
    conv34_up = concatenate([conv33,up31], axis =-1)
    conv35 = Conv3D(4*ini_channel, (3, 3, 3),activation=activationtype, padding='same', data_format = 'channels_last')(conv34_up)
       
    up32 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv35)
    conv35_up = concatenate([conv32,up32], axis =-1)
    conv36 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv35_up)
    
    up33 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv36)
    conv36_up = concatenate([conv31,up33], axis =-1)
    conv37 = Conv3D(ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv36_up)    
    
    conv38 = Conv3D(out_channel, (1, 1, 1), activation='relu',kernel_initializer = 'he_normal')(conv37)
    print("conv18:" ,conv38.shape)
    
    Wavg4 = Lambda(COM1)(conv38)
    List4 = Reshape((1,3))(Wavg4)
    out4 = Lambda(Computelayer)([output3,List4])
    fo = Lambda(Broadlayer)(out4)
    fo = Lambda(Addlayer)([fo,out4])
    model = Model(input=[inputs], output=[fo])
    
    #load weight
    model.load_weights(weight_path)
    model.compile(optimizer=Adam(lr=5e-4), loss=Root_MSE)
    return model

def Model_Predict_Evaluate(gpu):
    print('---Loading Test data...---')
    
    test_data = load_train_data()
        
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpu
      
    print('---Creating and compiling model...---')

    model = PointLoc(False, True ,activationtype = 'relu')
    
    model.load_weights(weight_path)

    print('---Predicting and Evaluating model...---')

    model_predict = model.predict(test_data, verbose =1 ,batch_size =1)
    np.savetxt('/.../Prediction0114.py', model_predict, delimiter = ',' )
    #model_evaluate= model.evaluate(test_data, test_mask, verbose =1, batch_size =1, )
    #np.savetxt('/home/tm478/bif/Evaluation0803c.py', model_evaluate, delimiter = ',' )

if __name__ == '__main__':
    Model_Predict_Evaluate("0")

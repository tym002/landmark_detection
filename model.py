import os
import sys
from Utils import printHeading
#import cv2
import numpy as np
from keras.models import Model
from keras.layers import Multiply,Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Dropout, AveragePooling3D, BatchNormalization,Lambda,Reshape,ZeroPadding3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import losses, regularizers
from keras.utils import plot_model
from keras import backend as K
from BBoxData_new import *
from skimage.transform import rotate,resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from patchlib import patch_gen
import pynd 
import tensorflow as tf
#from skimage import data
import matplotlib.pyplot as plt
import h5py
from keras.utils import multi_gpu_model
from MyCallback import *

#path of the pre-trained weight
weight_path = '/home/tm478/bif/fst_train0814a.hdf5'

tf.enable_eager_execution

#batch size 
b_size = 1
#number of outout channels 
out_channel = 1
#number of channels for the first conv layer
ini_channel = 16
#dropout rate
drop_rate = 0.5
#dimensions of the input image
img_rows = 448	
img_cols = 448
img_batch = 448
#original input size
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
    '''
    load the training data and ground truth coordinates at 4 scales 
    '''
    imgs_train = np.load('/home/tm478/bif/Save0814/Train_Images448_ori.npy')
    imgs_mask_train56 = np.load('/home/tm478/bif/Save0814/Train_Mask_Coordinates56_ori.npy')
    imgs_mask_train112 = np.load('/home/tm478/bif/Save0814/Train_Mask_Coordinates112_ori.npy')
    imgs_mask_train224 = np.load('/home/tm478/bif/Save0814/Train_Mask_Coordinates224_ori.npy')
    imgs_mask_train448 = np.load('/home/tm478/bif/Save0814/Train_Mask_Coordinates448_ori.npy')
    return imgs_train, imgs_mask_train56, imgs_mask_train112,imgs_mask_train224, imgs_mask_train448 

def Resize(x):
    '''
    Resize takes an image tensor x as input and downsample 1/2 using 
    bilinear interpolation
    
    Arguments: 
        x is the input tensor with format [batch,x,y,z,channel]
    '''
    print("running resize function ...")
    print("the input tensor has shape: ",x.shape)
    x1 = tf.squeeze(x,[4])
    print("after squeeze: ",x1.shape)
    x2 = tf.image.resize_bilinear(x1,(int(img_rows/2),int(img_cols/2)))
    print("after resize: ",x2.shape)
    x3 = tf.transpose(x2,[0,1,3,2])
    print("after transpose: ",x3.shape)
    x4 = tf.image.resize_bilinear(x3,(int(img_rows/2),int(img_batch/2)))
    x5 = tf.transpose(x4,[0,1,3,2])
    x6 = tf.expand_dims(x5,4)
    print("new tensor has shape: ",x6.shape)
    return x6

def Resize1(x,p):
    '''
    Resize1 takes an image tensor x as input and downsample to 1/p using 
    skip pixel
    
    Arguments: 
        x is the input tensor with format [batch,x,y,z,channel]
        p is the downsample factor
    '''
    x1 = x[:,0::p,0::p,0::p,:]
    print("after resizing: ",x1.shape)
    return x1

def COM(feature,nx,ny,nz):
    '''
    COM computes the center of mass of the input image 
    
    Arguments: 
        feature: input image tensor with format [batch,x,y,z,channel]
        nx,ny,nz: dimensions of the input image
    '''
    map1 = feature
    x = K.sum(map1, axis =(2,3))

    r1 = tf.range(0,nx, dtype = 'float32')
    r1 = K.reshape(r1, (1,nx,1))
    
    x_product = x*r1
    x_weight_sum = K.sum(x_product,axis = 1,keepdims=True)+0.00001
    x_sum = K.sum(x,axis = 1,keepdims=True)+0.00001
    cm_x = tf.divide(x_weight_sum,x_sum)
    
    y = K.sum(map1, axis =(1,3))

    r2 = tf.range(0,ny, dtype = 'float32')
    r2 = K.reshape(r2, (1,ny,1))
    
    y_product = y*r2
    y_weight_sum = K.sum(y_product,axis = 1,keepdims=True)+0.00001
    y_sum = K.sum(y,axis = 1,keepdims=True)+0.00001
    cm_y = tf.divide(y_weight_sum,y_sum)
    
    z = K.sum(map1, axis =(1,2))

    r3 = tf.range(0,nz, dtype = 'float32')
    r3 = K.reshape(r3, (1,nz,1))
    
    z_product = z*r3
    z_weight_sum = K.sum(z_product,axis = 1,keepdims=True)+0.00001
    z_sum = K.sum(z,axis = 1,keepdims=True)+0.00001
    cm_z = tf.divide(z_weight_sum,z_sum)

    center_mass = tf.concat([cm_x,cm_y,cm_z],axis=1)

    return center_mass

def Padlayer(x,nx,ny,nz):
    '''
    Padlayer computes the number of pixels needed to pad the image 
    
    Arguments:
        x: input image tensor with format [batch,x,y,z,channel]
        nx,ny,nz: dimensions that want to pad the image to 
    '''
    print("before cropping: ",K.shape(x))
    image_rows = K.shape(x)[1]
    image_cols = K.shape(x)[2]
    image_batch = K.shape(x)[3]
    difsizeZ = nz - image_batch
    difsizeY = ny - image_cols
    difsizeX = nx - image_rows
    Vy = tf.cond(K.greater_equal(difsizeZ,0),lambda: K.cast(difsizeZ,'int32'),lambda:0)
    Xy = tf.cond(K.greater_equal(difsizeX,0),lambda: K.cast(difsizeX,'int32'),lambda:0)
    Yy = tf.cond(K.greater_equal(difsizeY,0),lambda: K.cast(difsizeY,'int32'),lambda:0)
    return K.reshape(K.stack([Xy,Yy,Vy]),(1,3))

def Croplayer(x,p):
    '''
    Croplayer crops the image to a patch of size (cx,cy,cz) in a differentiable
    manner by translating the image center to the predicted point using 
    biliear interpolation
    
    Arguments:
        x[0]: predicted point coordinates
        x[1]: image to be cropped
        b: dimensions of the input image
    '''
    pred,img = x
    mx = (p-1)/2
    my = (p-1)/2
    mz = (p-1)/2
    img1 = tf.squeeze(img,[4])
    l1 = mx - pred[0,0,0]*2
    l2 = my - pred[0,0,1]*2
    l3 = mz - pred[0,0,2]*2
    shift = tf.contrib.image.translate(img1,[l1,l2],interpolation='BILINEAR')
    x1 = tf.transpose(shift,[0,1,3,2])
    x2 = tf.contrib.image.translate(x1,[0,l3],interpolation='BILINEAR')   
    x3 = tf.transpose(x2,[0,1,3,2])
    x4 = tf.expand_dims(x3,4)
    unpadl = x4[:, K.maximum(0,int(p/2 - cx/2)) : K.minimum(p,int(p/2 + cx/2)), K.maximum(0, int(p/2-cy/2)) : K.minimum(p,int(p/2 + cy/2)) , K.maximum(0,int(p/2 - cz/2)) : K.minimum(p,int(p/2 + cz/2)),:]
    result1 = K.cast(unpadl,'float32')
    return result1

def Croplayer1(x,b):
    '''
    Croplayer1 crops the image to a patch of size (cx,cy,cz), with rounding
    
    Arguments:
        x[0]: predicted point coordinates
        x[1]: image to be cropped
        b: dimensions of the input image
    '''
    pred,img = x
    l1 =K.cast(pred[0,0,0],'int32')*2
    l2 = K.cast(pred[0,0,1],'int32')*2
    l3 =K.cast(pred[0,0,2],'int32')*2
    unpadl = img[:, K.maximum(0,l1-int(cx/2)) : K.minimum(b,l1+int(cx/2)), K.maximum(0,l2-int(cy/2)) : K.minimum(b,l2+int(cy/2)) , K.maximum(0,l3-int(cz/2)) : K.minimum(b,l3+int(cz/2)),:]
    result1 = K.cast(unpadl,'float32')
    return result1

def Computelayer(x):
    '''
    Computelayer computes the predicted coordinates on the original image. The 
    operation is not differentiable due to rounding
    
    Arguments:
        x[0]: prediction from previous stage
        x[1]: prediction from the current patch 
    '''
    pred1,pred2 = x
    l1 = K.cast(pred1[0,0,0],'int32')*2
    l2 = K.cast(pred1[0,0,1],'int32')*2
    l3 = K.cast(pred1[0,0,2],'int32')*2
    o1 = K.maximum(0,l1-int(cx/2))
    o2 = K.maximum(0,l2-int(cy/2))
    o3 = K.maximum(0,l3-int(cz/2))
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

def Computelayer1(x):
    '''
    Computelayer1 computes the predicted coordinates on the original image. The 
    operation is differentiable 
    
    Arguments:
        x[0]: prediction from previous stage
        x[1]: prediction from the current patch 
    '''
    pred1,pred2 = x
    l1 = pred1[0,0,0]*2
    l2 = pred1[0,0,1]*2
    l3 = pred1[0,0,2]*2
    o1 = K.maximum(0.0,l1-lx/2)
    o2 = K.maximum(0.0,l2-ly/2)
    o3 = K.maximum(0.0,l3-lz/2)
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

def Rotat(x):
    '''
    Rotat rotates the image with arbitrary angle
    
    Arguments:
        x[0]: input image to be rotated
        x[1]: rotation angle in degree
    '''
    inp,angle = x
    angle = K.cast(angle,'float32')
    x1 = tf.squeeze(inp,[4])
    rot = tf.contrib.image.rotate(x1,angle[0,0]*0.0174533,'BILINEAR')
    x2 = tf.expand_dims(rot,4)
    print("rotating image...")
    return x2

def Rannum(x):
    '''
    Rannum returns a uniformly distributed random number for any given interval
    '''
    angle = K.random_uniform((1,1), 10, 350)
    return angle

def rotate_around_point(p):
    """
    Rotate a point around a given point. 
    rotate_around_point returns the new coordinates after the rotation
    
    Arguments:
    p[0]: the coordinates of the original image. 
    p[1]: rotation angle 
    """
    pred,angle = p
    angle = 360 - angle[0,0]
    x1 = pred[0,0,0]
    y1 = pred[0,0,1]
    z1 = pred[0,0,2]
    x2 = pred[0,0,3]
    y2 = pred[0,0,4]
    z2 = pred[0,0,5]
    size = 448
    radians = -0.0174533*K.cast(angle,'float32')
    offset_x = (size - 1.0)/2
    offset_y = (size - 1.0)/2
    adjusted_x1 = (x1 - offset_x)
    adjusted_y1 = (y1 - offset_y)
    adjusted_x2 = (x2 - offset_x)
    adjusted_y2 = (y2 - offset_y)
    cos_rad = tf.math.cos(radians)
    sin_rad = tf.math.sin(radians)
    qx1 = offset_x + cos_rad * adjusted_x1 + sin_rad * adjusted_y1
    qy1 = offset_y + -sin_rad * adjusted_x1 + cos_rad * adjusted_y1
    qx2 = offset_x + cos_rad * adjusted_x2 + sin_rad * adjusted_y2
    qy2 = offset_y + -sin_rad * adjusted_x2 + cos_rad * adjusted_y2
    q = [qx1,qy1,z1,qx2,qy2,z2]
    q = K.stack(q)
    q = K.reshape(q,(1,3*out_channel))
     
    return q

def Ranshift(p):
    '''
    Ranshift return a randomly shifted point coordinates 
    
    Argument:
        p: the original point coordinates
    '''
    l = p[0,:,:]
    x = K.random_uniform((1,3), 0, 5) + l
    rs = K.reshape(x,(1,3))
    return rs
    
def Root_MSE(y_true, y_pred):
    '''
    sum of the Euclidean distance square between ground truth and prediction
    '''
    return K.sum(K.square((y_pred - y_true[:,:,0:3]))) 

def R_Square(y_true, y_pred):
    '''
    R_Square returns the R_square value given ground truth and the prediction
    
    '''
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def PointLoc(IsBN, Drop_1, activationtype):
    '''
    Architecture of the model 
    
    Arguments:
        IsBN: whether to use batch normalization 
        Drop_1: dropout rate
        activationtype: activation function, usually ReLU
    '''
    inputs = Input((img_rows, img_cols,img_batch,1))
    print(type(inputs))
    reduce1 = Lambda(Resize1,arguments={'p':8})(inputs)
    reduce2 = Lambda(Resize1,arguments={'p':4})(inputs)
    reduce3 = Lambda(Resize1,arguments={'p':2})(inputs)

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
    
    Wavg = Lambda(COM,arguments={'nx':cx,'ny':cy,'nz':cz})(conv8)
    output1 = Reshape((1,3),name = "Output1")(Wavg)
    #rs1 = Lambda(Ranshift)(output1)
    #rs1 = Reshape((1,3))(rs1)
    cropimg1 = Lambda(Croplayer1,arguments={'b':112})([output1,reduce2])
    s1 = Lambda(Padlayer,arguments={'nx':cx,'ny':cy,'nz':cz})(cropimg1)
    
    padimg1 = Lambda(K.spatial_3d_padding,arguments={'padding':((0,s1[0,0]),(0,s1[0,1]),(0,s1[0,2]))})(cropimg1)
    conv11 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(padimg1)
    if Drop_1:
        conv11 = Dropout(drop_rate)(conv11)
    pool11 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv11)


    conv12 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool11)   
    if Drop_1:
        conv12 = Dropout(drop_rate)(conv12)
    pool12 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv12)


    conv13 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool12)    
    if Drop_1:
        conv13 = Dropout(drop_rate)(conv13)
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
    
    Wavg2 = Lambda(COM,arguments={'nx':cx,'ny':cy,'nz':cz})(conv18)
    List2 = Reshape((1,3))(Wavg2)
    out2 = Lambda(Computelayer1)([output1,List2])
    output2 = Reshape((1,3),name = "Output2")(out2)
    #rs2 = Lambda(Ranshift)(output2)
    #rs2 = Reshape((1,3))(rs2)
    cropimg2 = Lambda(Croplayer1,arguments={'b':224})([output2,reduce3])
    s2 = Lambda(Padlayer,arguments={'nx':cx,'ny':cy,'nz':cz})(cropimg2)
    
    padimg2 = Lambda(K.spatial_3d_padding,arguments={'padding':((0,s2[0,0]),(0,s2[0,1]),(0,s2[0,2]))})(cropimg2)
    conv21 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(padimg2)
    if Drop_1:
        conv21 = Dropout(drop_rate)(conv21)
    pool21 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv21)

    conv22 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool21)   
    if Drop_1:
        conv22 = Dropout(drop_rate)(conv22)
    pool22 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv22)

    conv23 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool22)    
    if Drop_1:
        conv23 = Dropout(drop_rate)(conv23)
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
    
    Wavg3 = Lambda(COM,arguments={'nx':cx,'ny':cy,'nz':cz})(conv28)
    List3 = Reshape((1,3))(Wavg3)
    out3 = Lambda(Computelayer1)([output2,List3])
    output3 = Reshape((1,3),name = "Output3")(out3)
    rs3 = Lambda(Ranshift)(output3)
    rs3 = Reshape((1,3))(rs3)
    cropimg3 = Lambda(Croplayer1,arguments={'b':448})([rs3,inputs])
    s3 = Lambda(Padlayer,arguments={'nx':lx,'ny':ly,'nz':lz})(cropimg3)
    padimg3 = Lambda(K.spatial_3d_padding,arguments={'padding':((0,s3[0,0]),(0,s3[0,1]),(0,s3[0,2]))})(cropimg3)

    conv31 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(padimg3)
    if Drop_1:
        conv31 = Dropout(drop_rate)(conv31)
    pool31 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv31)

    conv32 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool31)   
    if Drop_1:
        conv32 = Dropout(drop_rate)(conv32)
    pool32 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv32)

    conv33 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool32)    
    if Drop_1:
        conv33 = Dropout(drop_rate)(conv33)
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
    
    Wavg4 = Lambda(COM,arguments={'nx':lx,'ny':ly,'nz':lz})(conv38)
    List4 = Reshape((1,3))(Wavg4)
    out4 = Lambda(Computelayer1)([rs3,List4])
    output4 = Reshape((1,3),name = "Output4")(out4)
    
    #model = Model(input=[inputs], output=[output1,output2,output3,output4])
    model = Model(input=[inputs], output=[output4])
    #los = {"Output1" : Root_MSE,"Output2": Root_MSE, "Output3" : Root_MSE,"Output4": Root_MSE}
    #lossesWeight = {"Output1" : 16,"Output2": 4,"Output3" : 1,"Output4": 0.25 }
    

    #model.compile(optimizer=Adam(lr=5e-4), loss= los, loss_weights=lossesWeight)
    #model.compile(optimizer=Adam(lr=5e-4), loss= Root_MSE)
    return model

def TrainandValidate(gpu):  

    printHeading('Loading and preprocessing train data...')
    
    imgs_train, imgs_mask_train56, imgs_mask_train112,imgs_mask_train224,imgs_mask_train448 = load_train_data()
        
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpu
      
    printHeading('Creating and compiling model...')

    model = PointLoc(False, True ,activationtype = 'relu')
    
    #save the model
    model_checkpoint = ModelCheckpoint('fst_train0122.hdf5', monitor='val_loss',verbose=1, save_best_only=True,save_weights_only = True)
    
    printHeading('Fitting model...')
    #alpha = K.variable(16)
    #beta = K.variable(4)
    #gamma = K.variable(1)
    #delta = K.variable(0.25)
    #los = {"Output1": Root_MSE,"Output2": Root_MSE,"Output3":Root_MSE, "Output4": Root_MSE}
    #lossesWeight = {"Output1":1, "Output2": 1,"Output3": 1, "Output4": 1 }
    #lossesWeight = {"Output1":alpha, "Output2": beta,"Output3": gamma, "Output4": delta }
    #model.load_weights(weight_path)
    #model.compile(optimizer=Adam(lr=5e-4), loss= los, loss_weights=lossesWeight)
    model.compile(optimizer=Adam(lr=5e-4), loss= Root_MSE)
    
    #mtrain= model.fit(imgs_train, {"Output1" :imgs_mask_train56 , "Output2": imgs_mask_train112,"Output3" :imgs_mask_train224 , "Output4": imgs_mask_train448 }, batch_size= b_size, validation_split = 0.2 , epochs=500, verbose=1, shuffle=True,callbacks=[model_checkpoint]) 
    #mtrain= model.fit(imgs_train, {"Output1" :imgs_mask_train56 , "Output2": imgs_mask_train112,"Output3" :imgs_mask_train224 , "Output4": imgs_mask_train448 }, batch_size= b_size, validation_split = 0.2 , epochs=500, verbose=1, shuffle=True,callbacks=[model_checkpoint,MyCallback(alpha,beta,gamma,delta)]) 
    mtrain= model.fit(imgs_train, imgs_mask_train448, batch_size= b_size, validation_split = 0.2 , epochs=200, verbose=1, shuffle=True,callbacks=[model_checkpoint])

    plt.plot(mtrain.history['loss'])
    plt.plot(mtrain.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'upper left')
    plt.show()
    plt.savefig('TrainingDrop0122')
    
    plt.plot(mtrain.history['acc'])
    plt.plot(mtrain.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'upper left')
    plt.show()
    plt.savefig('AccuracyDrop0122')
    
    plot_model(mtrain, to_file='model.png')
if __name__ == '__main__':
    model = TrainandValidate("0")
  

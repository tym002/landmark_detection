import os
import sys
from Utils import printHeading
#import cv2
import numpy as np
from keras.models import Model
from keras.layers import Multiply,Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Dropout, AveragePooling3D, BatchNormalization,Lambda
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import losses, regularizers
from keras.utils import plot_model
from keras import backend as K
from skimage.transform import rotate,resize, downscale_local_mean
from sklearn.model_selection import train_test_split
from patchlib import patch_gen
import pynd 
import tensorflow as tf
from CenterOfMass_ntrain import *
#from skimage import data
import matplotlib.pyplot as plt

#path of the pre-trained weight
weight_path = '/home/tm478/bif/fst_train0810c.hdf5'

#batch size 
b_size = 1
#number of outout channels 
out_channel = 1
#number of channels for the first conv layer
ini_channel = 16
#dropout rate
drop_rate = 0.5

#dimensions of the input image
img_rows = 208	
img_cols = 208
img_batch = 208

def load_train_data():
    '''
    load the training data and its label
    '''
    imgs_train = np.load('/home/tm478/bif/Save0608/Train_Images208_ori_process1.npy')
    imgs_mask_train = np.load('/home/tm478/bif/Save0608/Train_gaussian208_24_ori.npy')
    return imgs_train, imgs_mask_train

def soft_dice_loss(y_true, y_pred): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
    '''
    
    # skip the batch and class axis for calculating Dice score 
    numerator = 2. * K.sum(y_pred * y_true)
    denominator = K.sum(K.square(y_pred) + K.square(y_true))
    loss = 1 - (numerator / (denominator + 0.000001))
    return loss

def Root_MSE(y_true, y_pred):
    '''
    Root_MSE calculate the Euclidean distance square between y_true and y_pred
    '''
    return K.mean(K.sum(K.square(y_pred - y_true))) 

def R_Square(y_true, y_pred):
	SS_res = K.sum(K.square(y_true[:,:,0:3] - y_pred))
	SS_tot = K.sum(K.square(y_true[:,:,0:3] - K.mean(y_true)))
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
    conv1 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(inputs)
    if IsBN:
        conv1 = BatchNormalization(axis =-1, momentum =0.99)(conv1)
    #conv1 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer = 'lecun_uniform', data_format = 'channels_last')(conv1)
    if Drop_1:
        conv1 = Dropout(drop_rate)(conv1)
    if IsBN:
        conv1 = BatchNormalization(axis =-1, momentum =0.99)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv1)

    conv2 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool1)
    if IsBN:
        conv2 = BatchNormalization(axis =-1, momentum =0.99)(conv2)
    #conv2 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'lecun_uniform', data_format = 'channels_last')(conv2)
    if Drop_1:
        conv2 = Dropout(drop_rate)(conv2)
    if IsBN:
        conv2 = BatchNormalization(axis =-1, momentum =0.99)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv2)

    conv3 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool2)
    if IsBN:
        conv3 = BatchNormalization(axis =-1, momentum =0.99)(conv3)
    #conv3 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'lecun_uniform', data_format = 'channels_last')(conv3)
    if Drop_1:
        conv3 = Dropout(drop_rate)(conv3)
    if IsBN:
        conv3 = BatchNormalization(axis =-1, momentum =0.99)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv3)

    conv4 = Conv3D(8*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool3)
    if IsBN:
        conv4 = BatchNormalization(axis =-1, momentum =0.99)(conv4)
    #conv4 = Conv3D(8*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'lecun_uniform', data_format = 'channels_last')(conv4)
    if Drop_1:
        conv4 = Dropout(drop_rate)(conv4)
    if IsBN:
        conv4 = BatchNormalization(axis =-1, momentum =0.99)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), data_format = 'channels_last')(conv4)
    
    conv5 = Conv3D(16*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(pool4)
    if IsBN:
        conv5 = BatchNormalization(axis =-1, momentum =0.99)(conv5)
    #conv5 = Conv3D(16*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'lecun_uniform', data_format = 'channels_last')(conv5)
    if Drop_1:
        conv5 = Dropout(drop_rate)(conv5)

    up1 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv5)
    conv5_up = concatenate([conv4,up1], axis =-1)
    conv6 = Conv3D(8*ini_channel, (3, 3, 3),activation=activationtype, padding='same', kernel_initializer = 'he_normal',data_format = 'channels_last')(conv5_up)
    if Drop_1:
        conv6 = Dropout(drop_rate)(conv6)
    #conv6 = Conv3D(8*ini_channel, (3, 3, 3),activation=activationtype, padding='same', kernel_initializer = 'lecun_uniform',data_format = 'channels_last')(conv6)
   
    up2 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv6)
    conv6_up = concatenate([conv3,up2], axis =-1)
    conv7 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv6_up)
    if Drop_1:
        conv7 = Dropout(drop_rate)(conv7)
    #conv7 = Conv3D(4*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'lecun_uniform', data_format = 'channels_last')(conv7)
    
    up3 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv7)
    conv7_up = concatenate([conv2,up3], axis =-1)
    conv8 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv7_up)
    if Drop_1:
        conv8 = Dropout(drop_rate)(conv8)
    #conv8 = Conv3D(2*ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'lecun_uniform', data_format = 'channels_last')(conv8)
    
    
    up4 = UpSampling3D(size=(2, 2 ,2), data_format = 'channels_last')(conv8)
    conv8_up = concatenate([conv1,up4], axis =-1)
    conv9 = Conv3D(ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'he_normal', data_format = 'channels_last')(conv8_up)
    if Drop_1:
        conv9 = Dropout(drop_rate)(conv9)
    #conv9 = Conv3D(ini_channel, (3,3, 3), activation=activationtype, padding='same', kernel_initializer = 'lecun_uniform', data_format = 'channels_last')(conv9)
    
    conv10 = Conv3D(out_channel, (1, 1, 1), activation=activationtype, kernel_initializer = 'he_normal')(conv9)
    print("conv10:" ,conv10.shape)
    
    #Wavg = CenterofMass(out_channel,3)(conv10)
    
    model = Model(input=[inputs], output=[conv10])
    
    #load weight
    #model.load_weights(weight_path)
    model.compile(optimizer=Adam(lr=1e-6), loss= 'mean_squared_error', metrics = [soft_dice_loss])
    return model

# Train and ploting. 
def TrainandValidate(gpu):  

    printHeading('Loading and preprocessing train data...')
    
    imgs_train, imgs_mask_train = load_train_data()
        
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpu
      
    printHeading('Creating and compiling model...')

    model = PointLoc(False, True ,activationtype = 'relu')
    
    #save the model
    model_checkpoint = ModelCheckpoint('fst_train0827d.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    
    printHeading('Fitting model...')
    
    mtrain= model.fit(imgs_train, imgs_mask_train, batch_size= b_size, validation_split = 0.2 , epochs=1000, verbose=1, shuffle=True,callbacks=[model_checkpoint]) 

    plt.plot(mtrain.history['loss'])
    plt.plot(mtrain.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'upper left')
    plt.show()
    plt.savefig('TrainingDrop0827d')
    
    plt.plot(mtrain.history['acc'])
    plt.plot(mtrain.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'upper left')
    plt.show()
    plt.savefig('AccuracyDrop0602c')
    
    plot_model(mtrain, to_file='model.png')
if __name__ == '__main__':
    model = TrainandValidate("1")
    

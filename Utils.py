#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:57:49 2019

@author: Amrut
"""
import numpy as np 

def printHeading(header):
    '''
    print a header
    '''
    print('-'*30)
    print(header)
    print('-'*30)

#shape the image by padding and then croping
def img_resize(max_batch, image_rows, image_cols,image_batch, img):
    padded_img =zeropad(max_batch, image_rows, image_cols,image_batch, img)
    return img_crop(max_batch, padded_img)

def zeropad(max_batch, image_rows, image_cols,image_batch, img):
    '''
    pad the image with zero on far side of the image
    '''
    difsizeZ = max_batch - image_batch
    difsizeY = max_batch - image_cols
    difsizeX = max_batch - image_rows
    #print(difsizeZ,difsizeY,difsizeX)
    if difsizeZ >= 0:
        Vx = 0
        Vy = int(difsizeZ)
    else: 
        Vx = 0
        Vy = 0
    if difsizeX >= 0:
        Xx = 0
        Xy = int(difsizeX)
    else: 
        Xx = 0
        Xy = 0
    if difsizeY >= 0:
        Yx = 0
        Yy = int(difsizeY)
    else: 
        Yx = 0
        Yy = 0
    img = np.pad(img, ((Xx,Xy),(Yx,Yy),(Vx,Vy)), 'constant')
    print(img.shape)
    return img

def img_crop(max_batch, img):
    '''
    crop the img on the far side according to max_batch
    '''
    image_rows, image_cols,image_batch = img.shape
    if max_batch < image_rows:
        dif_x = int(image_rows - max_batch)
        crop_x = int(dif_x/ 2)
        crop_x1 =int(dif_x - crop_x)
        img = img[0:512,:,:]
    if max_batch < image_cols:
        dif_y = int(image_cols - max_batch)
        crop_y = int(dif_y/ 2)
        crop_y1 = int(dif_y - crop_y)
        img = img[:,0:512,:]
    if max_batch < image_batch:
        dif_z = int(image_batch - max_batch)
        print("Img batch and dif: ",image_batch," ", dif_z)
        crop_z = int(dif_z/ 2)
        print("cropZ: ", crop_z)
        crop_z1 = int(dif_z - crop_z) 
        img = img[:,:,0:512]
    return img
    
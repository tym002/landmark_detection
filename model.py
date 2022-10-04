import os
import numpy as np
import argparse
import tensorflow as tf
import pandas as pd
from keras.models import Model
from Utils import print_heading
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Dropout, BatchNormalization, Lambda, \
    Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from MyCallback import *

# path of the pre-trained weight
weight_path = '/home/tm478/bif/fst_train0814a.hdf5'

# batch size
b_size = 1
# number of outout channels
out_channel = 1
# number of channels for the first conv layer
ini_channel = 16
# dropout rate
drop_rate = 0.5
# dimensions of the input image
img_rows = 448
img_cols = 448
img_batch = 448
# original input size
osize = 448
# input patch size
cx = 56
cy = 56
cz = 56

# input patch size for the last Localizer Net
lx = 56
ly = 56
lz = 56


def load_train_data():
    """
    load the training data and ground truth coordinates at 4 scales
    """
    imgs_train = np.load('data/train/Train_Images448_ori.npy')
    imgs_mask_train56 = np.load('data/train/Train_Mask_Coordinates56.npy')
    imgs_mask_train112 = np.load('data/train/Train_Mask_Coordinates112.npy')
    imgs_mask_train224 = np.load('data/train/Train_Mask_Coordinates224.npy')
    imgs_mask_train448 = np.load('data/train/Train_Mask_Coordinates448.npy')
    return imgs_train, imgs_mask_train56, imgs_mask_train112, imgs_mask_train224, imgs_mask_train448


def load_test_data():
    img_test = np.load('data/test/Test_Images448_ori_p.npy')
    return img_test


def Resize(x):
    """
    Resize takes an image tensor x as input and downsample 1/2 using
    bilinear interpolation

    Arguments:
        x is the input tensor with format [batch,x,y,z,channel]
    """
    print("running resize function ...")
    print("the input tensor has shape: ", x.shape)
    x1 = tf.squeeze(x, [4])
    print("after squeeze: ", x1.shape)
    x2 = tf.image.resize_bilinear(x1, (int(img_rows / 2), int(img_cols / 2)))
    print("after resize: ", x2.shape)
    x3 = tf.transpose(x2, [0, 1, 3, 2])
    print("after transpose: ", x3.shape)
    x4 = tf.image.resize_bilinear(x3, (int(img_rows / 2), int(img_batch / 2)))
    x5 = tf.transpose(x4, [0, 1, 3, 2])
    x6 = tf.expand_dims(x5, 4)
    print("new tensor has shape: ", x6.shape)
    return x6


def Resize1(x, p):
    """
    Resize1 takes an image tensor x as input and downsample to 1/p using
    skip pixel

    Arguments:
        x is the input tensor with format [batch,x,y,z,channel]
        p is the downsample factor
    """
    x1 = x[:, 0::p, 0::p, 0::p, :]
    print("after resizing: ", x1.shape)
    return x1


def COM(feature, nx, ny, nz):
    """
    COM computes the center of mass of the input image

    Arguments:
        feature: input image tensor with format [batch,x,y,z,channel]
        nx,ny,nz: dimensions of the input image
    """
    map1 = feature
    x = K.sum(map1, axis=(2, 3))

    r1 = tf.range(0, nx, dtype='float32')
    r1 = K.reshape(r1, (1, nx, 1))

    x_product = x * r1
    x_weight_sum = K.sum(x_product, axis=1, keepdims=True) + 0.00001
    x_sum = K.sum(x, axis=1, keepdims=True) + 0.00001
    cm_x = tf.divide(x_weight_sum, x_sum)

    y = K.sum(map1, axis=(1, 3))

    r2 = tf.range(0, ny, dtype='float32')
    r2 = K.reshape(r2, (1, ny, 1))

    y_product = y * r2
    y_weight_sum = K.sum(y_product, axis=1, keepdims=True) + 0.00001
    y_sum = K.sum(y, axis=1, keepdims=True) + 0.00001
    cm_y = tf.divide(y_weight_sum, y_sum)

    z = K.sum(map1, axis=(1, 2))

    r3 = tf.range(0, nz, dtype='float32')
    r3 = K.reshape(r3, (1, nz, 1))

    z_product = z * r3
    z_weight_sum = K.sum(z_product, axis=1, keepdims=True) + 0.00001
    z_sum = K.sum(z, axis=1, keepdims=True) + 0.00001
    cm_z = tf.divide(z_weight_sum, z_sum)

    center_mass = tf.concat([cm_x, cm_y, cm_z], axis=1)

    return center_mass


def Padlayer(x, nx, ny, nz):
    """
    Padlayer computes the number of pixels needed to pad the image

    Arguments:
        x: input image tensor with format [batch,x,y,z,channel]
        nx,ny,nz: dimensions that want to pad the image to
    """
    print("before cropping: ", K.shape(x))
    image_rows = K.shape(x)[1]
    image_cols = K.shape(x)[2]
    image_batch = K.shape(x)[3]
    difsizeZ = nz - image_batch
    difsizeY = ny - image_cols
    difsizeX = nx - image_rows
    Vy = tf.cond(K.greater_equal(difsizeZ, 0), lambda: K.cast(difsizeZ, 'int32'), lambda: 0)
    Xy = tf.cond(K.greater_equal(difsizeX, 0), lambda: K.cast(difsizeX, 'int32'), lambda: 0)
    Yy = tf.cond(K.greater_equal(difsizeY, 0), lambda: K.cast(difsizeY, 'int32'), lambda: 0)
    return K.reshape(K.stack([Xy, Yy, Vy]), (1, 3))


def Croplayer(x, p):
    """
    Croplayer crops the image to a patch of size (cx,cy,cz) in a differentiable
    manner by translating the image center to the predicted point using
    biliear interpolation

    Arguments:
        x[0]: predicted point coordinates
        x[1]: image to be cropped
        b: dimensions of the input image
    """
    pred, img = x
    mx = (p - 1) / 2
    my = (p - 1) / 2
    mz = (p - 1) / 2
    img1 = tf.squeeze(img, [4])
    l1 = mx - pred[0, 0, 0] * 2
    l2 = my - pred[0, 0, 1] * 2
    l3 = mz - pred[0, 0, 2] * 2
    shift = tf.contrib.image.translate(img1, [l1, l2], interpolation='BILINEAR')
    x1 = tf.transpose(shift, [0, 1, 3, 2])
    x2 = tf.contrib.image.translate(x1, [0, l3], interpolation='BILINEAR')
    x3 = tf.transpose(x2, [0, 1, 3, 2])
    x4 = tf.expand_dims(x3, 4)
    unpadl = x4[:, K.maximum(0, int(p / 2 - cx / 2)): K.minimum(p, int(p / 2 + cx / 2)),
             K.maximum(0, int(p / 2 - cy / 2)): K.minimum(p, int(p / 2 + cy / 2)),
             K.maximum(0, int(p / 2 - cz / 2)): K.minimum(p, int(p / 2 + cz / 2)), :]
    result1 = K.cast(unpadl, 'float32')
    return result1


def Croplayer1(x, b):
    """
    Croplayer1 crops the image to a patch of size (cx,cy,cz), with rounding

    Arguments:
        x[0]: predicted point coordinates
        x[1]: image to be cropped
        b: dimensions of the input image
    """
    pred, img = x
    l1 = K.cast(pred[0, 0, 0], 'int32') * 2
    l2 = K.cast(pred[0, 0, 1], 'int32') * 2
    l3 = K.cast(pred[0, 0, 2], 'int32') * 2
    unpadl = img[:, K.maximum(0, l1 - int(cx / 2)): K.minimum(b, l1 + int(cx / 2)),
             K.maximum(0, l2 - int(cy / 2)): K.minimum(b, l2 + int(cy / 2)),
             K.maximum(0, l3 - int(cz / 2)): K.minimum(b, l3 + int(cz / 2)), :]
    result1 = K.cast(unpadl, 'float32')
    return result1


def Computelayer(x):
    """
    Computelayer computes the predicted coordinates on the original image. The
    operation is not differentiable due to rounding

    Arguments:
        x[0]: prediction from previous stage
        x[1]: prediction from the current patch
    """
    pred1, pred2 = x
    l1 = K.cast(pred1[0, 0, 0], 'int32') * 2
    l2 = K.cast(pred1[0, 0, 1], 'int32') * 2
    l3 = K.cast(pred1[0, 0, 2], 'int32') * 2
    o1 = K.maximum(0, l1 - int(cx / 2))
    o2 = K.maximum(0, l2 - int(cy / 2))
    o3 = K.maximum(0, l3 - int(cz / 2))
    x1 = pred2[0, 0, 0]
    x2 = pred2[0, 0, 1]
    x3 = pred2[0, 0, 2]
    X = x1 + K.cast(o1, 'float32')
    Y = x2 + K.cast(o2, 'float32')
    Z = x3 + K.cast(o3, 'float32')
    cor = [X, Y, Z]
    cor = K.stack(cor)
    cor = K.reshape(cor, (1, 3))
    return cor


def Computelayer1(x):
    """
    Computelayer1 computes the predicted coordinates on the original image. The
    operation is differentiable

    Arguments:
        x[0]: prediction from previous stage
        x[1]: prediction from the current patch
    """
    pred1, pred2 = x
    l1 = pred1[0, 0, 0] * 2
    l2 = pred1[0, 0, 1] * 2
    l3 = pred1[0, 0, 2] * 2
    o1 = K.maximum(0.0, l1 - lx / 2)
    o2 = K.maximum(0.0, l2 - ly / 2)
    o3 = K.maximum(0.0, l3 - lz / 2)
    x1 = pred2[0, 0, 0]
    x2 = pred2[0, 0, 1]
    x3 = pred2[0, 0, 2]
    X = x1 + K.cast(o1, 'float32')
    Y = x2 + K.cast(o2, 'float32')
    Z = x3 + K.cast(o3, 'float32')
    cor = [X, Y, Z]
    cor = K.stack(cor)
    cor = K.reshape(cor, (1, 3))
    return cor


def Rotat(x):
    """
    rotates the image with arbitrary angle

    Arguments:
        x[0]: input image to be rotated
        x[1]: rotation angle in degree
    """
    inp, angle = x
    angle = K.cast(angle, 'float32')
    x1 = tf.squeeze(inp, [4])
    rot = tf.contrib.image.rotate(x1, angle[0, 0] * 0.0174533, 'BILINEAR')
    x2 = tf.expand_dims(rot, 4)
    print("rotating image...")
    return x2


def Rannum(x):
    """
    returns a uniformly distributed random number for any given interval
    """
    angle = K.random_uniform((1, 1), 10, 350)
    return angle


def rotate_around_point(p):
    """
    Rotate a point around a given point. 
    rotate_around_point returns the new coordinates after the rotation
    
    Arguments:
    p[0]: the coordinates of the original image. 
    p[1]: rotation angle 
    """
    pred, angle = p
    angle = 360 - angle[0, 0]
    x1 = pred[0, 0, 0]
    y1 = pred[0, 0, 1]
    z1 = pred[0, 0, 2]
    x2 = pred[0, 0, 3]
    y2 = pred[0, 0, 4]
    z2 = pred[0, 0, 5]
    size = 448
    radians = -0.0174533 * K.cast(angle, 'float32')
    offset_x = (size - 1.0) / 2
    offset_y = (size - 1.0) / 2
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
    q = [qx1, qy1, z1, qx2, qy2, z2]
    q = K.stack(q)
    q = K.reshape(q, (1, 3 * out_channel))

    return q


def Ranshift(p):
    """
    return a randomly shifted point coordinates

    Argument:
        p: the original point coordinates
    """
    l = p[0]
    x = K.random_uniform((1, 3), 0, 5) + l
    rs = K.reshape(x, (1, 3))
    return rs


def Root_MSE(y_true, y_pred):
    """
    sum of the Euclidean distance square between ground truth and prediction
    """
    return K.sum(K.square((y_pred - y_true[..., 0:3])))


def R_Square(y_true, y_pred):
    """
    R_Square returns the R_square value given ground truth and the prediction

    """
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


def conv_layer_down(layer, IsBN, Drop_1, activationtype, channel):
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last')(layer)
    if IsBN:
        pool1 = BatchNormalization(axis=-1, momentum=0.99)(pool1)
    if Drop_1:
        pool1 = Dropout(drop_rate)(pool1)
    conv1 = Conv3D(channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer='he_normal',
                   data_format='channels_last')(pool1)

    return conv1


def conv_layer_up(layer, layer_copy, activationtype, Drop_1):
    up1 = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(layer)
    conv_up = concatenate([up1, layer_copy], axis=-1)
    conv = Conv3D(4 * ini_channel, (3, 3, 3), activation=activationtype, padding='same', data_format='channels_last')(
        conv_up)
    if Drop_1:
        conv = Dropout(drop_rate)(conv)
    return conv


def unet(input, IsBN, Drop_1, activationtype, random_shift):
    conv1 = Conv3D(ini_channel, (3, 3, 3), activation=activationtype, padding='same', kernel_initializer='he_normal',
                   data_format='channels_last')(input)

    conv2 = conv_layer_down(conv1, IsBN, Drop_1, activationtype, 2 * ini_channel)

    conv3 = conv_layer_down(conv2, IsBN, Drop_1, activationtype, 4 * ini_channel)

    conv4 = conv_layer_down(conv3, IsBN, Drop_1, activationtype, 8 * ini_channel)

    conv5 = conv_layer_up(conv4, conv3, activationtype, Drop_1)

    conv6 = conv_layer_up(conv5, conv2, activationtype, Drop_1)

    conv7 = conv_layer_up(conv6, conv1, activationtype, Drop_1)

    conv8 = Conv3D(out_channel, (1, 1, 1), activation='relu', kernel_initializer='he_normal')(conv7)
    print("final conv layer shape:", conv8.shape)

    Wavg = Lambda(COM, arguments={'nx': cx, 'ny': cy, 'nz': cz})(conv8)
    output1 = Reshape((1, 3), name="Output1")(Wavg)
    if random_shift:
        output1 = Lambda(Ranshift)(output1)
        output1 = Reshape((1, 3))(output1)

    return output1


def crop_pad(output_coordinate, img, cx, cy, cz):
    cropimg1 = Lambda(Croplayer1, arguments={'b': int(cx * 2)})([output_coordinate, img])
    s1 = Lambda(Padlayer, arguments={'nx': cx, 'ny': cy, 'nz': cz})(cropimg1)

    padimg1 = Lambda(K.spatial_3d_padding, arguments={'padding': ((0, s1[0, 0]), (0, s1[0, 1]), (0, s1[0, 2]))})(
        cropimg1)
    return padimg1


def point_loc(IsBN, Drop_1, activationtype, multi_loss, random_shift):
    """
    Architecture of the model

    Arguments:
        :param IsBN: whether to use batch normalization
        :param Drop_1: dropout rate
        :param activationtype: activation function, usually ReLU
        :param random_shift: whether apply random shift after each key-point prediction before cropping
    """
    inputs = Input((img_rows, img_cols, img_batch, 1))

    reduce1 = Lambda(Resize1, arguments={'p': 8})(inputs)
    reduce2 = Lambda(Resize1, arguments={'p': 4})(inputs)
    reduce3 = Lambda(Resize1, arguments={'p': 2})(inputs)

    output1 = unet(reduce1, IsBN, Drop_1, activationtype, random_shift)
    padimg1 = crop_pad(output1, reduce2, cx, cy, cz)

    output2 = unet(padimg1, IsBN, Drop_1, activationtype, random_shift)
    padimg2 = crop_pad(output2, reduce3, cx, cy, cz)

    output3 = unet(padimg2, IsBN, Drop_1, activationtype, random_shift)
    padimg3 = crop_pad(output3, inputs, cx, cy, cz)

    output4 = unet(padimg3, IsBN, Drop_1, activationtype, random_shift)

    out4 = Lambda(Computelayer1)([output3, output4])
    output4 = Reshape((1, 3), name="Output4")(out4)

    if multi_loss:
        model = Model(input=[inputs], output=[output1, output2, output3, output4])
    else:
        model = Model(input=[inputs], output=[output4])

    return model


def train_validate(mode="train", gpu="0", load_weight=False, multi_loss=False, callback=False, random_shift=False):
    if mode not in ["train", "test"]:
        print("mode needs to be either train or test, exit without running")
        exit(0)
    save_folder = "results/"
    file_name = "train_result_2022"

    print_heading('Loading and preprocessing train data...')

    imgs_train, imgs_mask_train56, imgs_mask_train112, imgs_mask_train224, imgs_mask_train448 = load_train_data()
    img_test = load_test_data()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    print_heading('Creating and compiling model...')

    model = point_loc(False, True, 'relu', multi_loss, random_shift)
    print(model.summary())

    if load_weight or mode == "test":
        model.load_weights(weight_path)

    if mode == "train":
        # save the model
        model_checkpoint = ModelCheckpoint('fst_train.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                                           save_weights_only=True)

        print_heading('Fitting model...')

        if multi_loss:
            alpha = K.variable(16)
            beta = K.variable(4)
            gamma = K.variable(1)
            delta = K.variable(0.25)
            los = {"Output1": Root_MSE, "Output2": Root_MSE, "Output3": Root_MSE, "Output4": Root_MSE}
            loss_weight = {"Output1": alpha, "Output2": beta, "Output3": gamma, "Output4": delta}
            model.compile(optimizer=Adam(lr=5e-4), loss=los, loss_weights=loss_weight)
            if callback:
                mtrain = model.fit(imgs_train, {"Output1": imgs_mask_train56, "Output2": imgs_mask_train112,
                                                "Output3": imgs_mask_train224, "Output4": imgs_mask_train448},
                                   batch_size=b_size, validation_split=0.2, epochs=500, verbose=1, shuffle=True,
                                   callbacks=[model_checkpoint, MyCallback(alpha, beta, gamma, delta)])
            else:
                mtrain = model.fit(imgs_train, {"Output1": imgs_mask_train56, "Output2": imgs_mask_train112,
                                                "Output3": imgs_mask_train224, "Output4": imgs_mask_train448},
                                   batch_size=b_size, validation_split=0.2, epochs=500, verbose=1, shuffle=True,
                                   callbacks=[model_checkpoint])
        else:
            model.compile(optimizer=Adam(lr=5e-4), loss=Root_MSE)
            mtrain = model.fit(imgs_train, imgs_mask_train448, batch_size=b_size, validation_split=0.2, epochs=200,
                               verbose=1,
                               shuffle=True, callbacks=[model_checkpoint])

        pd.DataFrame.from_dict(mtrain.history).to_csv(save_folder + 'history_' + file_name + '.csv', index=False)
    else:
        model_predict = model.predict(img_test, verbose=1, batch_size=1)
        np.savetxt('results/Prediction.py', model_predict, delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--mode", default="train", help="train or test")
    parser.add_argument("--gpu", default="0", help="which gpu to use")
    parser.add_argument("--multi_loss", default=False, action="store_true",
                        help="whether to use multi-stage-loss schedule")
    parser.add_argument("--use_callback", default=False, action="store_true", help="whether to use loss fallback")
    parser.add_argument("--random_shift", default=False, action="store_true",
                        help="whether to use random-shift before cropping")

    args = parser.parse_args()
    train_validate(mode=args.mode, gpu=args.gpu, load_weight=False, multi_loss=args.multi_loss,
                   callback=args.use_callback,
                   random_shift=args.random_shift)

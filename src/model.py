from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, \
     Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D, \
     Conv2DTranspose, Average, Activation, Concatenate
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from utils.VARIABLES import *

def conv_bn(x, filters, kernel_size=3, strides=1):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def dilated_conv_bn(x, filters, kernel_size=3, rate=1):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               use_bias=True,
               dilation_rate=(rate,rate))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def dilated_spatial_pyramidal_pooling_conv(x, filters, kernel_size=3):
    x2 = Conv2D(filters=8,
                kernel_size=3,
                padding='same',
                use_bias=True,
                dilation_rate=(2,2))(x)
    x4 = Conv2D(filters=8,
                kernel_size=3,
                padding='same',
                use_bias=True,
                dilation_rate=(4,4))(x)
    x8 = Conv2D(filters=8,
                kernel_size=3,
                padding='same',
                use_bias=True,
                dilation_rate=(8,8))(x)
    x16 = Conv2D(filters=8,
                kernel_size=3,
                padding='same',
                use_bias=True,
                dilation_rate=(16,16))(x)
    
    x = Concatenate(axis=3)([x, x2, x4, x8, x16])
    x = BatchNormalization()(x)
    x = Conv2D(filters=filters,
               kernel_size=1,
               padding='same',
               use_bias=True)(x)
    x = Activation('relu')(x)
    return x

def model(x):
    # Encoder
    x = conv_bn(x, filters=16)
    x = conv_bn(x, filters=32)
    x1 = dilated_spatial_pyramidal_pooling_conv(x, filters=32)
    x = dilated_conv_bn(x, filters=64, rate=2)
    x = conv_bn(x, filters=32)
    x2 = dilated_spatial_pyramidal_pooling_conv(x, filters=32)
    x = dilated_conv_bn(x, filters=64, rate=4)
    x = conv_bn(x, filters=32)
    x4 = dilated_spatial_pyramidal_pooling_conv(x, filters=32)
    x = dilated_conv_bn(x, filters=64, rate=8)
    x = conv_bn(x, filters=32)
    x8 = dilated_spatial_pyramidal_pooling_conv(x, filters=32)
    x = dilated_conv_bn(x, filters=64, rate=16)
    x = conv_bn(x, filters=32)
    x = conv_bn(x, filters=32)

    # Decoder
    x = Concatenate(axis=3)([x8, x])
    x = conv_bn(x, filters=32)
    x = Concatenate(axis=3)([x4, x])
    x = conv_bn(x, filters=32)
    x = Concatenate(axis=3)([x2, x])
    x = conv_bn(x, filters=32)
    x = Concatenate(axis=3)([x1, x])
    x = conv_bn(x, filters=32)

    # Number of filters = number of classes
    x = Conv2D(filters=NUM_CLASSES, kernel_size=1, strides=1, activation='softmax')(x)
    return x
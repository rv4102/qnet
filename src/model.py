import tensorflow as tf
import tensorflow.keras.backend as K
from utils.VARIABLES import *

def conv_bn(x, filters, kernel_size=3, strides=1):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def dilated_conv_bn(x, filters, kernel_size=3, rate=1):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               use_bias=True,
                               dilation_rate=(rate,rate))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def dilated_spatial_pyramidal_pooling_conv(x, filters, kernel_size=3):
    x2 = tf.keras.layers.Conv2D(filters=8,
                                kernel_size=3,
                                padding='same',
                                use_bias=True,
                                dilation_rate=(2,2))(x)
    x4 = tf.keras.layers.Conv2D(filters=8,
                                kernel_size=3,
                                padding='same',
                                use_bias=True,
                                dilation_rate=(4,4))(x)
    x8 = tf.keras.layers.Conv2D(filters=8,
                                kernel_size=6,
                                padding='same',
                                use_bias=True,
                                dilation_rate=(8,8))(x)
    x16 = tf.keras.layers.Conv2D(filters=8,
                                 kernel_size=9,
                                 padding='same',
                                 use_bias=True,
                                 dilation_rate=(8,8))(x)
    
    x = tf.keras.layers.Concatenate(axis=3)([x, x2, x4, x8, x16])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def model(x):
    # Encoder
    x = conv_bn(x, filters=16)
    x = conv_bn(x, filters=32)
    x1 = dilated_spatial_pyramidal_pooling_conv(x, filters=16)
    x = dilated_conv_bn(x, filters=64, rate=2)
    x = conv_bn(x, filters=32)
    x2 = dilated_spatial_pyramidal_pooling_conv(x, filters=16)
    x = dilated_conv_bn(x, filters=64, rate=4)
    x = conv_bn(x, filters=32)
    x4 = dilated_spatial_pyramidal_pooling_conv(x, filters=16)
    x = dilated_conv_bn(x, filters=64, rate=8)
    x = conv_bn(x, filters=32)
    x8 = dilated_spatial_pyramidal_pooling_conv(x, filters=16)
    x = dilated_conv_bn(x, filters=64, rate=16)
    x = conv_bn(x, filters=32)
    x = conv_bn(x, filters=32)

    # Decoder
    x = tf.keras.layers.Concatenate(axis=3)([x8, x])
    x = conv_bn(x, filters=32)
    x = tf.keras.layers.Concatenate(axis=3)([x4, x])
    x = conv_bn(x, filters=32)
    x = tf.keras.layers.Concatenate(axis=3)([x2, x])
    x = conv_bn(x, filters=32)
    x = tf.keras.layers.Concatenate(axis=3)([x1, x])
    x = conv_bn(x, filters=32)

    # Number of filters = number of classes
    x = tf.keras.layers.Conv2D(filters=NUM_CLASSES, kernel_size=1, strides=1, activation='softmax')(x)
    return x
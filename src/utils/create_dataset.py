import tensorflow as tf
from glob import glob
import os
from utils.VARIABLES import *

def load_data(path):
    image_paths = sorted(glob(os.path.join(path, "images/*")))
    mask_paths = sorted(glob(os.path.join(path, "masks/*")))
    return image_paths, mask_paths

def load_image_and_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.per_image_standardization(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([TARGET_SIZE, TARGET_SIZE, 3])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    # # If we subtract 1 from mask,
    # # we ignore background in the one hot encoding 
    # # as tf.one_hot will convert -1 to [0, 0, 0, 0]
    # # and 0 as [1, 0, 0, 0]

    # mask = mask - 1 
    mask = tf.one_hot(mask, depth=NUM_CLASSES)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.squeeze(mask)
    mask.set_shape([TARGET_SIZE, TARGET_SIZE, NUM_CLASSES])
    return image, mask

def create_dataset(image_paths, mask_paths):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def test_train_val_split(full_dataset, size, train_percent=0.6, val_percent=0.2, batch_size=8):
    train_dataset = full_dataset.take(int(train_percent*size))
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = full_dataset.skip(int(train_percent*size))

    val_dataset = test_dataset.take(int(val_percent*size))
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.skip(int(val_percent*size))
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, val_dataset, test_dataset
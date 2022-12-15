import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

img_name = 'N-33-60-D-c-4-2'
IMAGE_PATH = '../input/landcoverai/images/{}.tif'.format(img_name)
LABEL_PATH = '../input/landcoverai/masks/{}.tif'.format(img_name)

def from_one_hot_to_rgb_bkup(class_indexes, palette=None):
    """ 
    https://stackoverflow.com/a/60811084/6328456
    Assign a different color to each class in the input tensor 
    """
    # 0 is background, 1 is building, 2 is woodland, 3 is water, 4 is road
    if palette is None:
        palette = tf.constant(
            [[255, 206, 92], #background - light orange
            [239, 67, 107], #building - bright red
            [5, 199, 147], #woodland - lime green
            [38, 84, 125], #water - dark moderate blue
            [255, 245, 235]] #road - white (pale orange)
        , dtype=tf.int32)

    H, W, _ = class_indexes.shape
    class_indexes = tf.cast(class_indexes, tf.int32)

    color_image = tf.gather(palette, class_indexes)
    color_image = tf.reshape(color_image, [H, W, 3])

    color_image = tf.cast(color_image, dtype=tf.float32)
    return color_image

def pad_image_to_tile_multiple(image3, tile_size, padding="CONSTANT"):
    '''
    https://stackoverflow.com/a/46181172/6328456
    Pad an image to a multiple of the tile size.
    Input:
        image3: A 3D tensor with shape (H,W,C)
        tile_size: Tuple denoting size of the tiles.
        padding: Padding type for tf.pad command.
    Returns:
        A padded tensor.
    '''
    original_image_size = image3.shape
    image_size = tf.shape(image3)[0:2]
    padding_ = tf.cast(tf.math.ceil(image_size / tile_size), tf.int32) * tile_size - image_size
    return original_image_size, tf.pad(image3, [[0, padding_[0]], [0, padding_[1]], [0, 0]], padding)

def remove_image_padding(image3, original_img_shape):
    image3 = image3[0:original_img_shape[0], 0:original_img_shape[1], :]
    return image3

def split_image(image3, tile_size):
    '''
    https://stackoverflow.com/a/46181172/6328456
    Split an image into tiles. Converts a 3D tensor with shape (H,W,C) to a 4D tensor with shape (B,H,W,C).
    Input:
        image3: A 3D tensor with shape (H,W,C)
        tile_size: Tuple denoting size of the tiles.
    Returns:
        A 4D tensor with shape (B,H,W,C)
    '''
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    return tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])

def unsplit_image(tiles4, image_shape):
    '''
    https://stackoverflow.com/a/46181172/6328456
    Unsplit a tiles into an image. Converts a 4D tensor with [B,H,W,C] to a 3D tensor with [H,W,C].
    Input:
        tiles4: A 4D tensor with shape (B,H,W,C)
        image_shape: Tuple denoting size of the image.
    Returns:
        A 3D tensor with shape (H,W,C)
    '''
    tile_width = tf.shape(tiles4)[1]
    serialized_tiles = tf.reshape(tiles4, [-1, image_shape[0], tile_width, image_shape[2]])
    rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2, 3])
    return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])

# model = tf.keras.models.load_model('../input/model6/model_6/', compile=False)

# Read image and convert to tensor
img = cv2.imread(IMAGE_PATH)
label = cv2.imread(LABEL_PATH, cv2.IMREAD_UNCHANGED)
img = tf.convert_to_tensor(img, tf.float32)
img = img / 255.0
label = tf.convert_to_tensor(label, tf.float32)
label = label[..., tf.newaxis]

# Pad the image and label
original_label_shape, _ = pad_image_to_tile_multiple(label, [256, 256])
padded_label_shape = _.shape
original_img_shape, img = pad_image_to_tile_multiple(img, [256, 256])

# Split the padded image into batches and pad the batches
tiles = split_image(img, [256, 256])

pred_masks = model.predict(tiles)
pred_masks = tf.math.argmax(pred_masks, axis=-1)
pred_masks = pred_masks[...,tf.newaxis]
pred_mask = unsplit_image(pred_masks, padded_label_shape)

# Remove the padding from the image and mask
img = remove_image_padding(img, original_img_shape)
pred_mask = remove_image_padding(pred_mask, original_label_shape)

# Convert the masks to a uint8 image
label = from_one_hot_to_rgb_bkup(label)
pred_mask = from_one_hot_to_rgb_bkup(pred_mask)
label = tf.keras.utils.array_to_img(label)
pred_mask = tf.keras.utils.array_to_img(pred_mask)

# Save as png
# img = tf.io.encode_png(tf.cast(img, tf.uint8))
# tf.io.write_file('{}_0.png'.format(img_name), img)
label.save('{}_label.png'.format(img_name))
pred_mask.save('{}_predMask.png'.format(img_name))
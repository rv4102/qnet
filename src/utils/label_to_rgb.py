import tensorflow as tf
import cv2

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

LABEL_PATH="/Users/rushilv/Documents/Summer_Internship_RemoteSensing/LandCover/output/masks/M-33-7-A-d-2-3_m_195.png"

label = cv2.imread(LABEL_PATH, cv2.IMREAD_UNCHANGED)
label = tf.convert_to_tensor(label, tf.float32)
label = label[..., tf.newaxis]
label = from_one_hot_to_rgb_bkup(label)
label = tf.keras.preprocessing.image.array_to_img(label)
label.save('{}_label.png'.format('M-33-7-A-d-2-3_195'))
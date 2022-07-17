import matplotlib.pyplot as plt
import tensorflow as tf
from utils.loss_functions import asym_unified_focal_loss

# model = tf.keras.models.load_model('../saved_models/', custom_objects={'loss_function':asym_unified_focal_loss})

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(mask):
    condensed_mask = tf.math.argmax(mask, axis=-1)
    condensed_mask = condensed_mask[..., tf.newaxis]
    return condensed_mask[0]

def show_predictions(model, dataset=None, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], create_mask(mask), create_mask(pred_mask)])
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix

################################
#         OneHot MeanIoU       #
################################
class OneHotMeanIoU(tf.keras.metrics.MeanIoU):
    '''
    Custom metric to calculate OneHotMeanIoU
    as the keras version is not available in tf 2.6
    '''
    def __init__(
        self,
        num_classes: int,
        name=None,
        dtype=None,
    ):
        super(OneHotMeanIoU, self).__init__(
            num_classes=num_classes,
            name=name,
            dtype=dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        # Select max hot-encoding channels to convert into all-class format
        y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        
        return super().update_state(y_true, y_pred, sample_weight)
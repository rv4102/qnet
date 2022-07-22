import numpy as np
import tensorflow as tf
from model import *
from utils.VARIABLES import *
from utils.visualize import show_predictions
from utils.create_dataset import *
from utils.metrics import OneHotMeanIoU
from utils.loss_functions import asym_unified_focal_loss

# Compile the dataset
image_paths, mask_paths = load_data(OUTPUT_DIR)
dataset = create_dataset(image_paths, mask_paths)
train, val, test = test_train_val_split(dataset, len(image_paths), train_percent, val_percent, BATCH_SIZE)

# Initialize the model
input = tf.keras.layers.Input(shape=[TARGET_SIZE, TARGET_SIZE, 3])
output = model(input)
model = tf.keras.Model(input, output)
# 0 is background, 1 is building, 2 is woodland, 3 is water, 4 is road
model.compile(optimizer='adam', loss=asym_unified_focal_loss(), 
              metrics=[OneHotMeanIoU(NUM_CLASSES, name='MeanIoU'),
                       tf.keras.metrics.Precision(name='bgrPcsn', class_id=0),
                       tf.keras.metrics.Precision(name='bldPcsn', class_id=1),
                       tf.keras.metrics.Precision(name='wldPcsn', class_id=2),
                       tf.keras.metrics.Precision(name='wtrPcsn', class_id=3),
                       tf.keras.metrics.Precision(name='roadPcsn', class_id=4),
                       tf.keras.metrics.Recall(name='bgrRcl', class_id=0),
                       tf.keras.metrics.Recall(name='bldRcl', class_id=1),
                       tf.keras.metrics.Recall(name='wldRcl', class_id=2),
                       tf.keras.metrics.Recall(name='wtrRcl', class_id=3),
                       tf.keras.metrics.Recall(name='roadRcl', class_id=4)
                      ])
# model.summary()

save_model_path = '/kaggle/working/checkpoint_3'

cp = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_model_path, 
    monitor='val_MeanIoU', 
    mode='max', 
    save_best_only=True
)
cb_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_MeanIoU',
    mode='max',
    min_delta=0.001,
    patience=4,
    verbose=1
)
cb_reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_MeanIoU',
    mode='min',
    factor=0.1,
    patience=4,
    verbose=1,
    min_lr=0.00001
)

# Train the model    
history = model.fit(train.repeat(), 
                    steps_per_epoch=int(np.ceil(train_percent*len(image_paths) / float(BATCH_SIZE))),
                    epochs=epochs,
                    validation_data=val.repeat(),
                    validation_steps=int(np.ceil(val_percent*len(image_paths) / float(BATCH_SIZE))),
                    callbacks=[cp, cb_earlystop, cb_reducelr],
                    verbose=1)

# Save model progress to file             
model.save('/kaggle/working/model_4')

# model = model = tf.keras.models.load_model('/Users/rushil/Documents/RemoteSensing/model_6/', custom_objects={'loss_function':asym_unified_focal_loss, 'MeanIoU':OneHotMeanIoU})

# Evaluate model on train dataset
loss, acc, *is_anything_else_being_returned = model.evaluate(test, verbose=1, batch_size=BATCH_SIZE)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# Print the top images from the test dataset
show_predictions(model, test, 15)
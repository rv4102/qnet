import numpy as np
import tensorflow as tf
from model import *
from utils.VARIABLES import *
from utils.visualize import display
from utils.create_dataset import *
from utils.metrics import precision, recall, OneHotMeanIoU
from utils.loss_functions import asym_unified_focal_loss

# Compile the dataset
image_paths, mask_paths = load_data(OUTPUT_DIR)
dataset = create_dataset(image_paths, mask_paths)
train, val, test = test_train_val_split(dataset, len(image_paths), train_percent, val_percent, BATCH_SIZE)

# Initialize the model
input = tf.keras.layers.Input(shape=[TARGET_SIZE, TARGET_SIZE, 3])
output = model(input)
model = tf.keras.Model(input, output)
model.compile(optimizer='adam', loss=asym_unified_focal_loss(), metrics=[OneHotMeanIoU(NUM_CLASSES, name='MeanIoU'), precision, recall, tf.keras.metrics.AUC(multi_label=True,
    num_labels=NUM_CLASSES, name='AUC')])
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

# Evaluate model on train dataset
loss, acc, *is_anything_else_being_returned = model.evaluate(test, verbose=1, batch_size=BATCH_SIZE)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# Print the top images from the test dataset
display(model, test, 15)
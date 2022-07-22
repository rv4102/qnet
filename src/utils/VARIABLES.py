# Variables
IMGS_DIR = "/Users/rushil/Documents/RemoteSensing/LandCover/images/"
MASKS_DIR = "/Users/rushil/Documents/RemoteSensing/LandCover/masks/"
OUTPUT_DIR = "/Users/rushil/Documents/RemoteSensing/LandCover/output/"

TARGET_SIZE = 256
NUM_CLASSES = 5
BATCH_SIZE = 20
epochs = 15
train_percent = 0.6
val_percent = 0.2
seed = 47

# 0 is background, 1 is building, 2 is woodland, 3 is water, 4 is road
class_weights = {0: 1/(1-(1.85+72.02+13.15+3.5)/216.27), 1: 1/(1.85/216.27), 2: 1/(72.02/216.27), 3: 1/(13.15/216.27), 4: 1/(3.5/216.27)}
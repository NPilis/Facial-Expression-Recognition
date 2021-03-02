from os import mkdir
from PIL import Image

def normalize_data(x):
    return x / 255.0

def resnet_preprocess(x, target_size=(197, 197), target_channels=3):
    x = cv2.resize(x, target_size, interpolation=cv2.INTER_CUBIC)
    x = np.repeat(x, repeats=target_channels, axis=-1)
    return x
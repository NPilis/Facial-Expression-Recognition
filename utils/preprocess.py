from os import mkdir
import cv2
import numpy as np

def resnet_preprocess(X_batch, target_size=(197, 197), target_channels=3):
    X = []
    for samp in X_batch:
        x = cv2.resize(x, target_size, interpolation=cv2.INTER_CUBIC) / 255
        x = np.expand_dims(x, axis=-1)
        x = np.repeat(x, repeats=target_channels, axis=-1)
        X.append(x)
    return np.array(X)
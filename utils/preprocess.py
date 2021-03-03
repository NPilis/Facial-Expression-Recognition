from os import mkdir
import cv2
import numpy as np


def resnet_preprocess(x_batch, target_size=(197, 197), target_channels=3):
    X = []
    for samp in x_batch:
        x = cv2.resize(samp, target_size, interpolation=cv2.INTER_NEAREST) / 255.0
        if x.shape[-1] != target_channels:
            x = np.expand_dims(x, axis=-1)
            x = np.repeat(x, repeats=target_channels, axis=-1)
        X.append(x)
    return np.array(X)

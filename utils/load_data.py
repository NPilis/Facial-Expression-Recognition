import os, cv2
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle

def load_FER2013(DATA_DIR):
    """ Load fer2013 dataset from csv file """

    data_path = DATA_DIR + '/fer2013/fer2013.csv'
    data = pd.read_csv(data_path)
    train = data[data['Usage'] == 'Training']
    val = data[data['Usage'] == 'PublicTest']
    test = data[data['Usage'] == 'PrivateTest']
    return train, val, test

def parse_FER2013(data, target_size=(48, 48), target_channel=1):
    """ Parse fer2013 data to 48x48 greyscale images,
        and one-hot vector as labels """
    real_image_size = (48, 48)
    real_image_channel = 1
    images = np.empty(shape=(len(data), *target_size, target_channel))
    labels = np.empty(shape=(len(data), 1))
    for i, idx in enumerate(data.index):
        img = np.fromstring(data.loc[idx, 'pixels'], dtype='uint8', sep=' ')
        img = np.reshape(img, (48, 48))
        if target_size != real_image_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        img = img[..., np.newaxis]
        if target_channel != real_image_channel:
            img = np.repeat(img, repeats=3, axis=-1)
        label = data.loc[idx, 'emotion']
        images[i] = img
        labels[i] = label
    labels = to_categorical(labels, 7)
    return images, labels
    
def load_CKPlus(DATA_DIR, emotion_label_map):
    """ Load CK+ dataset from directory """

    data_path = DATA_DIR + '/CK+48'
    labels = []
    images = []
    for dataset in os.listdir(data_path):
        for img in os.listdir(data_path + '/' + dataset):
            input_img = cv2.imread(data_path + '/' + dataset + '/' + img, 0)
            input_img = cv2.resize(input_img, (48,48))
            images.append(input_img)
            labels.append(emotion_label_map[dataset])
    images = np.array(images)
    images = images[..., np.newaxis]
    labels = np.array(labels)
    labels = to_categorical(labels, 7)
    images, labels = shuffle(images, labels)
    return images, labels
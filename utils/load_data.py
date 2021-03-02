import os, cv2
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle
from PIL import Image

def load_fer2013(file_path='../data/fer2013/fer2013.csv'):
    """ Load fer2013.csv dataset from csv file """
    df = pd.read_csv(file_path)
    train = df[df['Usage'] == 'Training']
    val = df[df['Usage'] == 'PublicTest']
    test = df[df['Usage'] == 'PrivateTest']
    return train, val, test

def parse_fer2013(data, target_size=(48, 48), target_channel=1):
    """ Parse fer2013 data to images with specified sizes,
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
            img = np.repeat(img, repeats=target_channel, axis=-1)
        label = data.loc[idx, 'emotion']
        images[i] = img
        labels[i] = label
    labels = to_categorical(labels, 7)
    return images, labels
    
def load_CKPlus(file_path='../data/CK+48'):
    """ Load CK+ dataset from directory """
    emotions = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3,
                'sadness': 4, 'surprise': 5, 'neutral': 6}
    labels, images = [], []
    for dataset in os.listdir(file_path):
        for img in os.listdir(file_path + '/' + dataset):
            input_img = cv2.imread(file_path + '/' + dataset + '/' + img, 0)
            input_img = cv2.resize(input_img, (48,48))
            images.append(input_img)
            labels.append(emotions[dataset])
    images = np.array(images)
    images = images[..., np.newaxis]
    labels = np.array(labels)
    labels = to_categorical(labels, 7)
    images, labels = shuffle(images, labels)
    return images, labels

def fer_csv_to_png(file_path='../data/fer2013/fer2013.csv', data_dir='../data/fer/'):
    """ Unpack fer2013.csv file and save images to
        specified folders """
    emotions = {0:'angry', 1:'disgust', 2:'fear',
                3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}
    df = pd.read_csv(file_path)
    for dataset in ("train", 'val', "test"):
        for emotion in emotions:
            os.mkdir(data_dir + f'{dataset}/{emotion} {emotions[emotion]}')
    count = 0
    for emotion, pixels, usage in zip(df['emotion'],df['pixels'],df['Usage']):
        img = np.fromstring(pixels, dtype='uint8', sep=' ').reshape(48, 48)
        img = Image.fromarray(img)
        count_string = str(count).zfill(6)
        path = data_dir
        if usage == 'Training':
            path += 'train/'
        elif usage == 'PublicTest':
            path += 'val/'
        else:
            path += 'test/'
        emo_label = emotions[emotion]
        img.save(path + f'{emotion} {emo_label}/{emo_label}-{count_string}.png')
        count += 1
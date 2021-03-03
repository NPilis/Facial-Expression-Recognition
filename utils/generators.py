from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from .preprocess import resnet_preprocess
import numpy as np
import cv2


def init_resnet_generator(data_dir='../data/fer/train', with_aug=True,
                          img_size=(197, 197), batch_size=128):
    """ Initialize resnet generator with specified data flow
        from directory. Minimum ResNet50 img_size - (197, 197) with
        rgb channel """
    if with_aug:
        data_generator = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255
        )
    else:
        data_generator = ImageDataGenerator(rescale=1./255)
    return data_generator.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size)


def init_cnn_generator(with_aug=True):
    """ Initialize simple cnn generator """
    if with_aug:
        data_generator = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=15,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255
        )
    else:
        data_generator = ImageDataGenerator(rescale=1./255)
    return data_generator


class ResizeImageGenerator(Sequence):
    """ Custom image generator which resizes images to
        specified size and generates batches.
        #TODO implement custom augumentation """

    def __init__(self, images, labels, batch_size=64, target_size=(197, 197),
                 n_channels=3, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x_batch = self.images[indexes]
        x_batch = resnet_preprocess(x_batch, target_size=self.target_size,
                                    target_channels=self.n_channels)
        y_batch = self.labels[indexes]
        return x_batch, y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

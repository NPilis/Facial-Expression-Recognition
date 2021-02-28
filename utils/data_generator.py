from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .preprocess import resnet_preprocess

def init_resnet_generator(with_aug=True):
    if with_aug:
        data_generator = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255,
            preprocessing_function=resnet_preprocess
        )
    else:
        data_generator = ImageDataGenerator(rescale=1./255)
    return data_generator

def init_cnn_generator(with_aug=True):
    if with_aug:
        data_generator = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255,
        )
    else:
        data_generator = ImageDataGenerator(rescale=1./255)
    return data_generator
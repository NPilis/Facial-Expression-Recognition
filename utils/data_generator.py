from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .preprocess import resnet_preprocess

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
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255
        )
    else:
        data_generator = ImageDataGenerator(rescale=1./255)
    return data_generator
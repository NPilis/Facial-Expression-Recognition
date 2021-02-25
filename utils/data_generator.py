from tensorflow.keras.preprocessing.image import ImageDataGenerator

def initialize_generator():
    data_generator = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255
    )
    return data_generator
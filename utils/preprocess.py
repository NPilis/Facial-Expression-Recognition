from os import mkdir
from PIL import Image

def normalize_data(x):
    return x / 255.0

def resnet_preprocess(x, target_size=(224, 224), target_channels=3):
    x = cv2.resize(x, target_size, interpolation=cv2.INTER_CUBIC)
    x = np.repeat(x, repeats=target_channels, axis=-1)
    return x

def fer_csv_to_png(df, data_dir):
    emotions = {0:'angry', 1:'disgust', 2:'fear',
                3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}
    for dataset in ("train", 'val', "test"):
        for emotion in emotions:
            mkdir(data_dir + f'{dataset}/{emotion} {emotions[emotion]}')
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
        emo = emotions[emotion]
        img.save(path + f'{emotion} {emo}/{emo}-{count_string}.png')
        count += 1
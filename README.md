# Facial expression recognition
Project made with python to predict human emotional expressions given images of people's faces using deep neural networks. The goal is to train model to accurately classify human face emotion from the image. This repository demonstrates several neural network architectures trained on **FER2013** dataset which constist of 48x48 images labelled on 7 emotions - (angry, disgust, fear, happy, sad, suprise, neutral)

## Table of contents

* [Neural network architectures](#neural-network-architectures)
* [Datasets](#datasets)
* [Data augumentation](#data-augumentation)
* [Technologies](#technologies)
* [Usage](#usage)
* [Future work](#future-work)
* [References](#references)

## Neural network architectures
1. Simple Baseline CNN model - [Link](#https://arxiv.org/abs/2004.11823)
2. Pre-trained ResNet50 using Keras VGG-Face - Input size of the the network can't be smaller than 197x197 and must have 3 channels. So additional resizing and expanding dimensions of images was needed during training.

## Datasets
- FER2013 can be downloaded on [Kaggle](#https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) - dataset with ~ 35,000 greyscale images with faces automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.

![Alt text](assets/gen_faces2.png?raw=true)

## Data augumentation
Data augumentation was used to improve model's performance. These techniques increase size of dataset by applying various transformations such as mirroring, cropping, shifting, and rotation.

![Alt text](assets/aug_faces.png?raw=true)

## Techologies
- Tensorflow
- Keras
- OpenCV
- NumPy
- PIL
- Pandas

## Usage
Install all required packages
```
pip install -r requirements.txt
```
In order to test models make sure to download all pre-trained weights from [saved_models/](models/saved_models) folder
### Webcam demo
```
python webcam.py --model_path <path to model>
```
### Training
Download **FER2013** dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) - extract *fer2013.tar.gz* to [data/](data/) folder.
To train neural networks you can use notebooks from [notebooks/](notebooks/) folder or adjust and run `train.py` script file
```
python models/train.py
```

## Future work
In addition to get better results of both model accuracy and execution time:

- Add different data distributrions
- Try out different NN architectures
- Use ensemble learning
- Use one NN for both detection and classification - this would reduce execution time significantly

## References
1. https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
2. https://arxiv.org/abs/2004.11823
3. https://arxiv.org/pdf/1804.08348.pdf

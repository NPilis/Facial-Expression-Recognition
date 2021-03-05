import sys
import argparse
import re
import cv2
import numpy as np
from keras.models import load_model


def detect_classify_display(frame, model_type='resnet'):
    emotions = ['angry', 'disgust', 'fear',
                'happy', 'sad', 'surprise', 'neutral']
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        face = frame_gray[y:y+h, x:x+w]
        if model_type == 'resnet50':
            face = cv2.resize(face, (197, 197)) / 255
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            face = np.repeat(face, 3, -1)
        else:
            face = cv2.resize(face, (48, 48)) / 255
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
        predictions = model.predict(face)
        pred = np.argmax(predictions)
        prob = predictions[0, pred] * 100
        frame = cv2.rectangle(
            frame, (x, y), (x + w, y + h), (255, 255, 255), 4)
        cv2.putText(frame, emotions[pred] + f'   {prob:.2f}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.imshow('Emotion recognition', frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam - detect classify display.')
    parser.add_argument('--model_path', help='Path to model.', default='models/saved_models/fer2013_resnet50_1-e41-a0.71.hdf5')
    args = parser.parse_args()

    # Loading model
    model_path = args.model_path
    model_type = re.search(r'(?<=fer2013_)(.*?)(?=_)', model_name)
    model = load_model(saved_models_path + model_name)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened:
        print('Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('No captured frame -- Break!')
            break
        detect_classify_display(frame, model_type[0])
        if cv2.waitKey(10) == 27:
            break

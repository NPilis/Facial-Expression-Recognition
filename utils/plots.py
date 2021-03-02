from matplotlib import pyplot as plt
import numpy as np

def plot_face_with_label(face, label, possible_emotions=None):
    fig, ax = plt.subplots()
    ax.imshow(face)
    if possible_emotions and not isinstance(label, str):
        label_idx = np.argmax(label)
        label = possible_emotions[label_idx]
    ax.set_title(label)

def plot_model_history(history):
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    train_loss = history['loss']
    val_loss = history['val_loss']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.plot(train_acc, label='Training Accuracy')
    ax1.plot(val_acc, label='Validation Accuracy')
    ax2.plot(train_loss, label='Training Loss')
    ax2.plot(val_loss, label='Validation Loss')
    ax1.set_title("Model accuracy")
    ax2.set_title("Model loss")
    ax1.legend()
    ax2.legend()

def plot_generated_images(data_gen, X_train):
    it = data_gen.flow(X_train, batch_size=1)
    for i in range(9):
        plt.subplot(331 + i)
        batch = it.next()
        image = batch[0]
        plt.imshow(image)
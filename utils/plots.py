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

def plot_generated_images(data_gen, n_epochs=3, b_size=6, figsize=(11,6)):
    emotions = {0:'angry', 1:'disgust', 2:'fear',
                3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}
    counter = 0  
    fig, axs = plt.subplots(n_epochs, b_size, figsize=figsize)
    for epoch in range(n_epochs):
        for (x_batch, y_batch) in data_gen:
            for i in range(b_size):
                emotion_label = emotions[np.argmax(y_batch[i])]
                axs[counter][i].imshow(x_batch[i], "gray")
                axs[counter][i].set_title(emotion_label)
                axs[counter][i].set_axis_off()
            break
        data_gen.on_epoch_end()
        counter += 1
    plt.show()
    fig.savefig('../assets/gen_faces_.png')

def plot_augumented_images(data_gen, n_epochs=3, b_size=6, figsize=(11,6)):
    counter = 1
    fig = plt.figure(figsize=figsize)
    for epoch in range(n_epochs):
        for i, (x_batch, _) in enumerate(data_gen):
            plt.subplot(b_size, n_epochs, counter)
            plt.imshow(x_batch[0], "gray")
            plt.axis('off')
            counter += 1
            if i == b_size - 1:
                break
    plt.show()
    fig.savefig('../assets/aug_faces.png')
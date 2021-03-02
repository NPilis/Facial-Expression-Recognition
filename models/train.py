import sys
import os

project_path = os.path.abspath(os.path.join('../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from keras.callbacks import ModelCheckpoint, EarlyStopping
from models import init_cnn_baseline_model, init_resnet_model
from utils.load_data import load_FER2013, parse_FER2013
from utils.data_generator import init_cnn_generator
from utils.preprocess import normalize_data
from utils.plots import plot_model_history

# Parameters
batch_size = 64
num_epochs = 300
patience = 40
dataset_path = '../data'
saved_models_path = 'saved_models/'
dataset_name = 'fer2013'
model_name = '_simple_CNN'

# Loading data
train_data, val_data, test_data = load_FER2013(dataset_path)
X_train, Y_train = parse_FER2013(train_data)
X_val, Y_val = parse_FER2013(val_data)
X_test, Y_test = parse_FER2013(test_data)

# Data generator
train_gen = init_cnn_generator(with_aug=True)
val_gen = init_cnn_generator(with_aug=False)

# Model init
model = init_cnn_baseline_model()
model.summary()

# Callbacks
model_path = saved_models_path + dataset_name + model_name
model_path += '-e{epoch:02d}-a{val_accuracy:.2f}.hdf5'
early_stop = EarlyStopping('val_loss', patience=patience)
model_checkpoint = ModelCheckpoint(model_path, save_best_only=True)
callbacks = [model_checkpoint, early_stop]

# Training
history = model.fit(train_gen.flow(X_train, Y_train, batch_size),
                    validation_data=val_gen.flow(X_val, Y_val, batch_size),
                    epochs=num_epochs,
                    callbacks=callbacks)

# Saving history callback
history_name = saved_models_path + 'history_' + dataset_name + model_name
best_acc = np.max(history.history['val_accuracy'])
cnt_epoch = len(history.history['val_accuracy'])
np.save(f'{history_name}-e{cnt_epoch:02d}-a{best_acc:.2f}.npy', history.history)

# Model evaluation on test data
loss, acc = model.evaluate(X_test, Y_test)
print("Model accuracy: {:5.2f}%".format(100 * acc))
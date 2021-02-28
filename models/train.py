import sys
import os

project_path = os.path.abspath(os.path.join('../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from keras.callbacks import ModelCheckpoint, EarlyStopping
from models import simple_CNN, mini_Xception
from utils.load_data import load_FER2013, parse_FER2013
from utils.data_generator import initialize_generator
from utils.preprocess import normalize_data
from utils.plots import plot_model_history

# Parameters
batch_size = 64
num_epochs = 300
patience = 40
dataset_path = '../data'
saved_models_path = 'saved_models/'
dataset_name = 'fer2013'

# Loading data
train_data, val_data, test_data = load_FER2013(dataset_path)
X_train, Y_train = parse_FER2013(train_data)
X_val, Y_val = parse_FER2013(val_data)
X_test, Y_test = parse_FER2013(test_data)

# Preprocess
X_val = normalize_data(X_val)
X_test = normalize_data(X_test)

# Data generator
data_generator = initialize_generator()

# Model init
model = simple_CNN()
model.summary()

# Callbacks
model_name = saved_models_path + dataset_name + '_simple_CNN'
model_name += '-e{epoch:02d}-a{val_accuracy:.2f}.hdf5'
early_stop = EarlyStopping('val_loss', patience=patience)
model_checkpoint = ModelCheckpoint(model_name, save_best_only=True)

callbacks = [model_checkpoint, early_stop]

history = model.fit(data_generator.flow(X_train, Y_train, batch_size),
                    epochs=num_epochs,
                    callbacks=callbacks,
                    validation_data=(X_val, Y_val))

test_preds = model.predict(X_test).argmax(axis=1)
acc = (test_preds == np.argmax(Y_test, axis=1)).mean()
loss, acc = model.evaluate(X_test, Y_test)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
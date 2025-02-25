import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Read data with train-test split
ds_train, ds_test = tfds.load("mnist", split=['train', 'test'],
                              shuffle_files=True, as_supervised=True)

# Set up BatchDataset from the OptionsDataset object
ds_train = ds_train.batch(32)
ds_test = ds_test.batch(32)

# Build LeNet5 model and fit
model = Sequential([
    Conv2D(6, (5,5), input_shape=(28,28,1), padding="same", activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(120, (5,5), activation="tanh"),
    Flatten(),
    Dense(84, activation="tanh"),
    Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["sparse_categorical_accuracy"])
earlystopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
model.fit(ds_train, validation_data=ds_test, epochs=100, callbacks=[earlystopping])

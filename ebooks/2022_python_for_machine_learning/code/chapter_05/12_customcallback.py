import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class EpochCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Starting epoch {}".format(epoch + 1))

    def on_epoch_end(self, epoch, logs=None):
        print("Finished epoch {}".format(epoch + 1))


def simple_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model


# Loading the MNIST training and testing data splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pre-processing the training data
x_train = x_train / 255.0
x_train = x_train.reshape(60000, 28, 28, 1)
y_train_cat = to_categorical(y_train, 10)

model = simple_model()
model.fit(x_train,
          y_train_cat,
          batch_size=32,
          epochs=5,
          callbacks=[EpochCallback()],
          verbose=0)

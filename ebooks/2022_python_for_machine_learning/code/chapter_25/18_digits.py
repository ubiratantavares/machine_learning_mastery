import dbm
import pickle
import random

import numpy as np
import sklearn.datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# get digits dataset (8x8 images of digits)
digits = sklearn.datasets.load_digits()

# create file if not exists, otherwise open for read/write
with dbm.open("digits.dbm", "c") as db:
    for idx in range(len(digits.target)):
        db[str(idx)] = pickle.dumps((digits.images[idx], digits.target[idx]))

# retrieving data from database for model
def datagen(batch_size):
    """A generator to produce samples from database
    """
    with dbm.open("digits.dbm", "r") as db:
        keys = db.keys()
        while True:
            images = []
            targets = []
            for key in random.sample(keys, batch_size):
                image, target = pickle.loads(db[key])
                images.append(image)
                targets.append(target)
            yield np.array(images).reshape(-1,64), np.array(targets)

# Classification model in Keras
model = Sequential()
model.add(Dense(32, input_dim=64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["sparse_categorical_accuracy"])

# Train with data from dbm store
history = model.fit(datagen(32), epochs=5, steps_per_epoch=1000)

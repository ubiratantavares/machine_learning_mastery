import numpy as np

sequence = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))

# define model
import tensorflow as tf
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Input
from tensorflow.keras import Sequential, Model

model = Sequential([
    LSTM(100, activation="relu", input_shape=(n_in+1, 1)),
    RepeatVector(n_in),
    LSTM(100, activation="relu", return_sequences=True),
    TimeDistributed(Dense(1))
])
model.compile(optimizer="adam", loss="mse")

model.fit(sequence, sequence, epochs=300, verbose=0)

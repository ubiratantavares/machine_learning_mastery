# -*- coding: utf-8 -*-

# exemplo cnn-lstm univariado

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# dividir uma sequência univariada em amostras
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# encontre o fim desse padrão
		end_ix = i + n_steps
		# verifique se estamos além da sequência
		if end_ix > len(sequence)-1:
			break
		# reunir partes de entrada e saída do padrão
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# definir sequência de entrada
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# escolha várias etapas de tempo
n_steps = 4

# dividir em amostras
X, y = split_sequence(raw_seq, n_steps)

# remodelar de [amostras, recursos] para [amostras, subsequencias, etapas do tempo, recursos]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

# definir modelo CNN-LSTM
model = Sequential()
model.add(TimeDistributed(Conv1D(64, 1, activation = 'relu'), input_shape = (None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D()))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')


# ajustar o modelo
model.fit(X, y, epochs = 500, verbose = 0)

# demonstrar a previsão
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose = 0)
print(yhat)
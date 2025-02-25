# -*- coding: utf-8 -*-

# exemplo lstm de codificador-decodificador de várias etapas univariado.
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed


# dividir uma sequência univariada em amostras
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# encontre o fim desse padrão
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# verifique se estamos além da sequência
		if out_end_ix > len(sequence):
			break
		# # reunir partes de entrada e saída do padrão
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# definir sequência de entrada
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# escolha várias etapas de tempo
n_steps_in, n_steps_out = 3, 2


# dividido em amostras
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# resumir os dados
for i in range(len(X)):
	print(X[i], y[i])

# remodelar de [amostras, etapas do tempo] para [amostras, etapas do tempo, recursos]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))

# definir modelo lstm empilhado
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

# ajustar o modelo
epocas = 1000
model.fit(X, y, epochs = epocas, verbose = 0)

# demonstrar previsão
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose = 0)
print(yhat)


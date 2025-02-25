# -*- coding: utf-8 -*-

# exemplo convlstm univariado
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D

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

# remodelar de [amostras, etapas de tempo] para [amostras, etapas do tempo, linhas, colunas, recursos]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))

# definir modelo
model = Sequential()
model.add(ConvLSTM2D(64, (1,2), activation = 'relu', input_shape = (n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')


# ajustar o modelo
model.fit(X, y, epochs = 500, verbose = 0)


# demonstrar previsão
x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose = 0)
print(yhat)
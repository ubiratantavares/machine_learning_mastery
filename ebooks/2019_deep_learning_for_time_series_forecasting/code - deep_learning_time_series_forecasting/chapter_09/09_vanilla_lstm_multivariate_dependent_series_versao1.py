# -*- coding: utf-8 -*-

# exemplo lstm multivariado
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import LSTM, Dense

# dividir uma sequência multivariada em amostras
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# encontre o fim desse padrão
		end_ix = i + n_steps
		# verifique se estamos além do conjunto de dados
		if end_ix > len(sequences):
			break
		# reunir partes de entrada e saída do padrão
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# definir sequência de entrada
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

# definir sequência de saída
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# converte a estrutura para [linhas, colunas]
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# empilhar colunas horizontalmente
dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)
print('\n')
print(dataset.shape)
print('\n')

# escolha uma série de etapas de tempo
n_steps = 3

# converter em entrada / saída
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
print('\n')

# resumir os dados
for i in range(len(X)):
	print(X[i], y[i])
print('\n')

# o conjunto de dados sabe o número de recursos, por exemplo 2
n_features = X.shape[2]

# definir o modelo lstm tradicional
model = Sequential()
model.add(LSTM(50, 
               activation = 'relu', 
               input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer = 'adam', 
              loss = 'mse')

epocas = 200
# ajustar o modelo
model.fit(X, y, epochs = epocas, verbose = 0)

# demonstrar previsão
x_input = array([[80, 85], [90, 95], [100, 105]]) #  nova entrada da série temporal multivariada
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose = 0)
print(yhat)

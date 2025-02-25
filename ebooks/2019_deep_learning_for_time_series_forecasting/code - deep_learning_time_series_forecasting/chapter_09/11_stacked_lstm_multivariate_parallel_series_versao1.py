# -*- coding: utf-8 -*-

# exemplo lstm empilhado de saída multivariada
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
		if end_ix > len(sequences)-1:
			break
		# reunir partes de entrada e saída do padrão
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# definir sequência de entrada
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

# definir sequência de saída
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# converter para estrutura [linhas, colunas]
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# empilhar colunas horizontalmente
dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)
print('\n')

# escolha várias etapas de tempo
n_steps = 3

# converter em entrada / saída
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
print('\n')

# resumir os dados
for i in range(len(X)):
	print(X[i], y[i])

# o conjunto de dados sabe o número de recursos, por exemplo 2
n_features = X.shape[2]

# define o modelo lstm empilhado
model = Sequential()
model.add(LSTM(100, 
               activation = 'relu', 
               return_sequences = True, 
               input_shape = (n_steps, n_features)))

model.add(LSTM(100, 
               activation = 'relu'))

model.add(Dense(n_features))

model.compile(optimizer = 'adam', loss = 'mse')

# ajusta o modelo lstm empilhado
epocas = 400
model.fit(X, y, epochs = 400, verbose = 0)

# demonstrar previsão
x_input = array([[70,75,145], 
                 [80,85,165], 
                 [90,95,185]])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose = 0)

print(yhat)
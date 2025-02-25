# -*- coding: utf-8 -*-

# preparação de dados em várias etapas
from numpy import array

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
# -*- coding: utf-8 -*-

# preparação univariada de dados

from numpy import array

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

# definir a sequência de entrada
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# escolha várias etapas de tempo
n_steps = 3

# dividir em amostras
X, y = split_sequence(raw_seq, n_steps)

# resumir os dados
for i in range(len(X)):
	print(X[i], y[i])
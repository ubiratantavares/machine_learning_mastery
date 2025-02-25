# -*- coding: utf-8 -*-

# transformar séries temporais univariadas em problema de aprendizado supervisionado

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


# definir séries temporais univariadas
series = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(series.shape)

# transformar em um problema de aprendizado supervisionado
X, y = split_sequence(series, 3)
print(X.shape, y.shape)

# mostre cada amostra
for i in range(len(X)):
	print(X[i], y[i])
    
    
   


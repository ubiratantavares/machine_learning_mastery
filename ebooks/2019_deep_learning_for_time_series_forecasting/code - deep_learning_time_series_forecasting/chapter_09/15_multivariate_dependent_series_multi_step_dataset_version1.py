# -*- coding: utf-8 -*-

# preparação multivariada de dados em várias etapas.
from numpy import array, hstack


# dividir uma sequência multivariada em amostras
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# encontre o fim desse padrão
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# verifique se estamos além do conjunto de dados
		if out_end_ix > len(sequences):
			break
		# reunir partes de entrada e saída do padrão
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
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

# escolha várias etapas de tempo.
n_steps_in, n_steps_out = 3, 2

# converter em entrada / saída
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)
print('\n')

# resumir os dados
for i in range(len(X)):
	print(X[i], y[i])
# -*- coding: utf-8 -*-

# preparação de dados multivariada
from numpy import array, hstack


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
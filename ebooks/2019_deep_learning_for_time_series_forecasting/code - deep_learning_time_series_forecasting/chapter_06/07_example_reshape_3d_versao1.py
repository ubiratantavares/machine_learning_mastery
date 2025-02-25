# -*- coding: utf-8 -*-

# exemplo de criação de uma matriz 3d de subsequências
from numpy import array

# definir o conjunto de dados
data = list()
n = 5000

for i in range(n):
	data.append([i+1, (i+1)*10])
    
data = array(data)

# deletar a coluna do tempo
data = data[:, 1]

# dividir em amostras (por exemplo, 5000/200 = 25)
samples = list()
length = 200

# ultrapassar os 5.000 em saltos de 200
for i in range(0,n,length):
	# agarrar de i a i + 200
	sample = data[i : i + length]
	samples.append(sample)
    
# converter lista de matrizes em matriz 2D
data = array(samples)
print(data.shape)

# remodelar a matriz 2D para [amostras, etapas do tempo, recursos]
data = data.reshape((data.shape[0], data.shape[1], 1))
# data = data.reshape((len(samples), length, 1))
print(data.shape)
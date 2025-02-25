# -*- coding: utf-8 -*-

# exemplo de divisão de uma sequência univariada em subsequências
from numpy import array

# definir o conjunto de dados
data = list()
n = 5000

for i in range(n):
	data.append([i+1, (i+1)*10])
    
data = array(data)

# deletar a coluna do tempo
data = data[:, 1]

# dividido em amostras (por exemplo, 5000/200 = 25)
samples = list()
length = 200


# ultrapassar os 5.000 em saltos de 200
for i in range(0, n, length):
	# agarrar de i a i + 200
	sample = data[i : i + length]
	samples.append(sample)
    
print(len(samples))
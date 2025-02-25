# -*- coding: utf-8 -*-

# exemplo que descarta a dimensão de tempo do conjunto de dados
from numpy import array

# definir o conjunto de dados
data = list()
n = 5000

for i in range(n):
	data.append([i+1, (i+1)*10])
    
data = array(data)

# deletar a coluna do tempo
data = data[:, 1]
print(data.shape)
# -*- coding: utf-8 -*-

# exemplo de definição de um conjunto de dados

from numpy import array

# definir conjunto de dados
data = list()
n = 5000

for i in range(n):
	data.append([i+1, (i+1)*10])
    
data = array(data)
print(data[:5, :])
print(data.shape)


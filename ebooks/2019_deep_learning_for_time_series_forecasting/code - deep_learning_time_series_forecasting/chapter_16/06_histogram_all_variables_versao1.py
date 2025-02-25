# -*- coding: utf-8 -*-

# gráficos de histograma para o conjunto de dados de uso de energia
from pandas import read_csv
from matplotlib import pyplot as plt

# carregar o novo arquivo
dataset = read_csv('household_power_consumption.csv', 
                   header = 0, 
                   infer_datetime_format = True, 
                   parse_dates = ['datetime'], 
                   index_col = ['datetime'])


# gráfico de histograma para cada variável
plt.figure()

for i in range(len(dataset.columns)):
	# criar subgráfico
	plt.subplot(len(dataset.columns), 1, i+1)
    
	# obter nome da variável
	name = dataset.columns[i]
    
	# criar histograma
	dataset[name].hist(bins=100)
    
	# definir titulo
	plt.title(name, y=0, loc='right')
    
	# desativar ticks para remover a desordem
	plt.yticks([])
	plt.xticks([])

plt.show()
# -*- coding: utf-8 -*-

# plotagens de linha para o conjunto de dados de uso de energia.
from pandas import read_csv
from matplotlib import pyplot as plt


# carregar o novo arquivo
dataset = read_csv('household_power_consumption.csv', 
                   header = 0, 
                   infer_datetime_format = True, 
                   parse_dates = ['datetime'], 
                   index_col=['datetime'])

# plotagem de linhas para cada variável.
plt.figure()
for i in range(len(dataset.columns)):
	# criar o subgráfico
	plt.subplot(len(dataset.columns), 1, i + 1)
    
	# obtenha o nome da variável
	name = dataset.columns[i]
    
	# dados de plotagem
	plt.plot(dataset[name])
    
	# definir título
	plt.title(name, y = 0)
    
	# desative os ticks para remover a desordem.
	plt.yticks([])
	plt.xticks([])

plt.show()
    

# -*- coding: utf-8 -*-

# gráficos de linhas mensais para o conjunto de dados de uso de energia.
from pandas import read_csv
from matplotlib import pyplot as plt


# carregar o novo arquivo
dataset = read_csv('household_power_consumption.csv', 
                   header = 0, 
                   infer_datetime_format = True, 
                   parse_dates=['datetime'], 
                   index_col=['datetime'])

# traçar a potência ativa para cada mês
months = [x for x in range(1, 13)]


plt.figure()
for i in range(len(months)):
	# preparar o subgráfico
	ax = plt.subplot(len(months), 1, i + 1)
    
	# determinar o mês para traçar
	month = '2007-' + str(months[i])
    
	# obtenha todas as observações do mês.
	result = dataset[month]
    
	# traçar a potência ativa do mês
	plt.plot(result['Global_active_power'])
    
	# adicione um título ao subgráfico
	plt.title(month, y=0, loc='left')
    
	# desativar ticks para remover a desordem
	plt.yticks([])
	plt.xticks([])
    
plt.show()
# -*- coding: utf-8 -*-

# plotagens de linha diárias para o conjunto de dados de uso de energia
from pandas import read_csv
from matplotlib import pyplot as plt

# carregar o novo arquivo
dataset = read_csv('household_power_consumption.csv', 
                   header = 0, 
                   infer_datetime_format = True, 
                   parse_dates=['datetime'], 
                   index_col=['datetime'])

# traçar a potência ativa para cada dia
days = [x for x in range(1, 20)]

plt.figure()

for i in range(len(days)):
	# preparar o subgráfico
	ax = plt.subplot(len(days), 1, i + 1)
    
	# determinar o dia para traçar
	day = '2007-01-' + str(days[i])
    
	# obtenha todas as observações para o dia.
	result = dataset[day]
    
	# traçar a potência ativa do dia.
	plt.plot(result['Global_active_power'])
    
	# adicione um título à subgráfico
	plt.title(day, y=0, loc='left', size = 6)
    
	# desativar ticks para remover a desordem
	plt.yticks([])
	plt.xticks([])

plt.show()
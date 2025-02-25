# -*- coding: utf-8 -*-

# plotagens de linhas anuais para o conjunto de dados de uso de energia.
from pandas import read_csv
from matplotlib import pyplot as plt

# carregue o novo arquivo.
dataset = read_csv('household_power_consumption.csv', 
                   header = 0, 
                   infer_datetime_format = True, 
                   parse_dates = ['datetime'], 
                   index_col = ['datetime'])

# traçar a energia ativa para cada ano.
years = ['2007', '2008', '2009', '2010']

plt.figure()
for i in range(len(years)):
	# preparar o subgráfico
	ax = plt.subplot(len(years), 1, i + 1)
    
	# determine o ano para traçar.
	year = years[i]
    
	# obtenha todas as observações para o ano.
	result = dataset[str(year)]
    
	# traçar a potência ativa para o ano.
	plt.plot(result['Global_active_power'])
    
	# adicione um título à subgráfico
	plt.title(str(year), y = 0, loc = 'left')
    
	# desativar ticks para remover a desordem
	plt.yticks([])
	plt.xticks([])

plt.show()
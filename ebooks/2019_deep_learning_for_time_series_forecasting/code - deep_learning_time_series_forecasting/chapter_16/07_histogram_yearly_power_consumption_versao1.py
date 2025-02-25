# -*- coding: utf-8 -*-

# plotagens anuais de histograma para o conjunto de dados de uso de energia
from pandas import read_csv
from matplotlib import pyplot as plt


# carregar o novo arquivo
dataset = read_csv('household_power_consumption.csv', 
                    header = 0, 
                    infer_datetime_format = True, 
                    parse_dates = ['datetime'], 
                    index_col = ['datetime'])

# traçar a potência ativa para cada ano
years = ['2007', '2008', '2009', '2010']

plt.figure()

for i in range(len(years)):
	# criar o subgráfico
	ax = plt.subplot(len(years), 1, i + 1)
    
	# determinar o ano para traçar
	year = years[i]
    
	# obtenha todas as observações do ano
	result = dataset[str(year)]
    
	# traçar a potência ativa para o ano
	result['Global_active_power'].hist(bins=100)
    
	# ampliar a distribuição
	ax.set_xlim(0, 5)
    
	# adicione um título ao subgráfico
	plt.title(str(year), y = 0, loc = 'right')
    
	# desativar ticks para remover a desordemr
	plt.yticks([])
	plt.xticks([])
    
plt.show()
# -*- coding: utf-8 -*-

# carregar e limpar dados de uso de energia

from numpy import nan
from pandas import read_csv

# carregar todos os dados
dataset = read_csv('household_power_consumption.txt', 
                    sep=';', 
                    header = 0, 
                    low_memory = False, 
                    infer_datetime_format = True, 
                    parse_dates={'datetime':[0,1]}, 
                    index_col=['datetime'])

# resumir
print(dataset.shape)
print(dataset.head())

# marcar todos os valores ausentes
dataset.replace('?', nan, inplace = True)

# adicione uma coluna para o restante da submedição
values = dataset.values.astype('float32')
dataset['Sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])

# salvar conjunto de dados atualizado
dataset.to_csv('household_power_consumption.csv')

# carregue o novo conjunto de dados
dataset = read_csv('household_power_consumption.csv', 
                    header = 0, 
                    infer_datetime_format = True, 
                    parse_dates=['datetime'], 
                    index_col=['datetime'])

# resumir
print(dataset.shape)
print(dataset.head())

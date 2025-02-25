# -*- coding: utf-8 -*-

# avalie o cnn-lstm para o conjunto de dados de vendas mensais de carros.

from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from matplotlib import pyplot as plt

# divida um conjunto de dados univariado em conjuntos de treinamento / teste.
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transformar lista em formato de aprendizado supervisionado.
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# sequência de entrada (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# sequência de previsão (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# junte tudo
	agg = concat(cols, axis=1)
    
	# deletar linhas com valores de NaN
	agg.dropna(inplace = True)
    
	return agg.values

# erro quadrático médio da raiz ou rmse.
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# ajustar o modelo
def model_fit(train, config):
	# descompactar configuração
	n_seq, n_steps, n_filters, n_kernel, n_nodes, n_epochs, n_batch = config
	n_input = n_seq * n_steps
    
	# preparar os dados
	data = series_to_supervised(train, n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
	train_x = train_x.reshape((train_x.shape[0], n_seq, n_steps, 1))
    
	# definir o modelo
	model = Sequential()
	model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation = 'relu', input_shape=(None,n_steps,1))))
	model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation = 'relu')))
	model.add(TimeDistributed(MaxPooling1D()))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(n_nodes, activation = 'relu'))
	model.add(Dense(n_nodes, activation = 'relu'))
	model.add(Dense(1))
	model.compile(loss = 'mse', optimizer = 'adam')
    
	# ajustar
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model

# previsão com um modelo de pré-ajuste
def model_predict(model, history, config):
	# descompactar configuração
	n_seq, n_steps, _, _, _, _, _ = config
	n_input = n_seq * n_steps
    
	# preparar dados
	x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, 1))
    
	# previsão
	yhat = model.predict(x_input, verbose=0)
	return yhat[0]

# validação direta para dados univariados
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# divisão do conjunto de dados
	train, test = train_test_split(data, n_test)
    
	# ajuste do modelo
	model = model_fit(train, cfg)
    
	# histórico de sementes com conjunto de dados de treinamento.
	history = [x for x in train]
    
	# passe cada etapa do tempo no conjunto de teste
	for i in range(len(test)):
		# ajuste o modelo e faça previsões para o histórico.
		yhat = model_predict(model, history, cfg)
        
		# previsão da loja na lista de previsões
		predictions.append(yhat)
        
		# adicione observação real ao histórico para o próximo loop
		history.append(test[i])
        
	# estimar erro de previsão
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error

# repita a avaliação de uma configuração
def repeat_evaluate(data, config, n_test, n_repeats=30):
	# ajustar e avaliar o modelo n vezes
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores

# resumir o desempenho do modelo.
def summarize_scores(name, scores):
	# imprima um resumo.
	scores_m, score_std = mean(scores), std(scores)
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
	# gráfico box and whisker
	plt.boxplot(scores)
	plt.show()
    
# funcao principal
def main():
    series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
    data = series.values
    # divisão dos dados
    n_test = 12
    # definir configuraçãio
    config = [3, 12, 64, 3, 100, 200, 100]
    
    # pesquisa em grade
    scores = repeat_evaluate(data, config, n_test)
    
    # resumir pontuações
    summarize_scores('cnn-lstm', scores)
    

main()
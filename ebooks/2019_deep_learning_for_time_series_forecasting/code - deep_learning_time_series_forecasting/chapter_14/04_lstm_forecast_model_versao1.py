# -*- coding: utf-8 -*-

# avalie lstm para o conjunto de dados mensal de vendas de carros.
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

# conjunto de dados de diferença.
def difference(data, interval):
	return [data[i] - data[i - interval] for i in range(interval, len(data))]

# ajustar o modelo
def model_fit(train, config):
	# descompacte a configuração.
	n_input, n_nodes, n_epochs, n_batch, n_diff = config
    
	# preparar os dados
	if n_diff > 0:
		train = difference(train, n_diff)
        
	data = series_to_supervised(train, n_input)
	train_x, train_y = data[:, :-1], data[:, -1]
	train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    
	# definir o modelo
	model = Sequential()
	model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, 1)))
	model.add(Dense(n_nodes, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	# ajustar
	model.fit(train_x, train_y, epochs = n_epochs, batch_size = n_batch, verbose = 0)
	return model

# previsão com um modelo de pré-ajuste
def model_predict(model, history, config):
	# descompacte a configuração.
	n_input, _, _, _, n_diff = config
    
	# preparar os dados
	correction = 0.0
	if n_diff > 0:
		correction = history[-n_diff]
		history = difference(history, n_diff)
	x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    
	# previsão
	yhat = model.predict(x_input, verbose=0)
	return correction + yhat[0]

# validação direta para dados univariados.
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
    
	# conjunto de dados dividido.
	train, test = train_test_split(data, n_test)
    
	# ajustar o modelo
	model = model_fit(train, cfg)
    
	# histórico de sementes com conjunto de dados de treinamento.
	history = [x for x in train]
    
	# passe cada etapa do tempo no conjunto de teste
	for i in range(len(test)):
		# ajuste o modelo e faça previsões para o histórico.
		yhat = model_predict(model, history, cfg)
        
		# previsão da loja na lista de previsões
		predictions.append(yhat)
        
		# adicione observação real ao histórico para o próximo loop.
		history.append(test[i])
        
	# estimar erro de previsão.
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error

# repita a avaliação de uma configuração.
def repeat_evaluate(data, config, n_test, n_repeats = 30):
	# ajuste e avalie o modelo n vezes.
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores

# summarize model performance
def summarize_scores(name, scores):
	# print a summary
	scores_m, score_std = mean(scores), std(scores)
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
	# box and whisker plot
	plt.boxplot(scores)
	plt.show()
    
# funcao principal
def main():
    series = read_csv('monthly-car-sales.csv', header = 0, index_col = 0)
    data = series.values
    
    # divisão dos dados
    n_test = 12
    
    # definir configuração
    config = [36, 50, 100, 100, 12]
    
    # pesquisa em grade
    scores = repeat_evaluate(data, config, n_test)
    
    # resumir pontuações
    summarize_scores('lstm', scores)
    
main()
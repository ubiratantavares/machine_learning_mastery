# -*- coding: utf-8 -*-

# previsão de persistência para o conjunto de dados mensal de vendas de carros

from math import sqrt
from numpy import median
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# dividir um conjunto de dados univariado em conjuntos de treinamento/teste
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# erro médio quadrático da raiz ou rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# conjunto de dados de diferença
def difference(data, interval):
	return [data[i] - data[i - interval] for i in range(interval, len(data))]

# ajustar o modelo
def model_fit(train, config):
	return None

# previsão com um modelo de pré-ajuste
def model_predict(model, history, config):
	values = list()
	for offset in config:
		values.append(history[-offset])
	return median(values)

# validação direta para dados univariados.
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error

# repita a avaliação de uma configuração
def repeat_evaluate(data, config, n_test, n_repeats = 30):
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return scores

# resumir o desempenho do modelo.
def summarize_scores(name, scores):
	# imprimir um resumo
	scores_m, score_std = mean(scores), std(scores)
	print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
    
	# gráfico de caixa e bigode
	plt.boxplot(scores)
	plt.show()
    
# funcao principal
def main():
    series = read_csv('monthly-car-sales.csv', header = 0, index_col = 0)
    data = series.values

    # divisão dos dados
    n_test = 12
    
    # definir configuração
    config = [12, 24, 36]

    # pesquisa em grade
    scores = repeat_evaluate(data, config, n_test)

    # resumir pontuações
    summarize_scores('persistência', scores)
    
main()
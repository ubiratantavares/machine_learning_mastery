# convlstm codificador / decodificador de várias etapas univariado para o 
# conjunto de dados de uso de energia
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D

# dividir um conjunto de dados univariado em conjuntos de treinamento / teste
def split_dataset(data):
	# dividido em semanas padrão
	train, test = data[1:-328], data[-328:-6]
    
	# reestruturar em janelas de dados semanais
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# avalie uma ou mais previsões semanais em relação aos valores esperados
def evaluate_forecasts(actual, predicted):
	scores = list()
    
	# calcular uma pontuação RMSE para cada dia
	for i in range(actual.shape[1]):
		# calcular o mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
        
		# calcular o rmse
		rmse = sqrt(mse)
        
		# armazenar
		scores.append(rmse)
        
	# calcular o RMSE geral
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# resumir pontuações
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# converter histórico em entradas e saídas
def to_supervised(train, n_input, n_out=7):
	# achatar dados
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    
	X, y = list(), list()
	in_start = 0
    
	# percorrer toda a história um passo de cada vez
	for _ in range(len(data)):
        
		# define o final da sequência de entrada
		in_end = in_start + n_input
		out_end = in_end + n_out
        
		# verifique se temos dados suficientes para esta instância
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
            
		# avançar um passo
		in_start += 1
	return array(X), array(y)

# treinar o modelo
def build_model(train, n_steps, n_length, n_input):
	# preparar dados
	train_x, train_y = to_supervised(train, n_input)
    
	# define parâmetros
	verbose, epochs, batch_size = 0, 20, 16
	n_features, n_outputs = train_x.shape[2], train_y.shape[1]
    
	# remodelar em subsequências [amostras, etapas do tempo, linhas, colunas, canais]
	train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
    
	# remodelar a saída em [amostras, etapas do tempo, recursos]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    
	# definir o modelo
	model = Sequential()
	model.add(ConvLSTM2D(64, (1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
    
	# ajuste de rede
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# faça uma previsão
def forecast(model, history, n_steps, n_length, n_input):
	# achatar dados
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    
	# recuperar últimas observações para dados de entrada
	input_x = data[-n_input:, 0]
    
	# remodelar em [amostras, etapas do tempo, linhas, colunas, canais]
	input_x = input_x.reshape((1, n_steps, 1, n_length, 1))
    
	# prever a próxima semana
	yhat = model.predict(input_x, verbose=0)
    
	# nós queremos apenas a previsão do vetor
	yhat = yhat[0]
	return yhat

# avaliar um único modelo
def evaluate_model(train, test, n_steps, n_length, n_input):
	# ajustar o modelo
	model = build_model(train, n_steps, n_length, n_input)
    
	# história é uma lista de dados semanais
	history = [x for x in train]
    
	# validação progressiva a cada semana
	predictions = list()
    
	for i in range(len(test)):
		# prever a semana
		yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
        
		# armazena as previsões
		predictions.append(yhat_sequence)
        
		# obter observação real e adicionar ao histórico para prever a próxima semana
		history.append(test[i, :])
        
	# avaliar dias de previsões para cada semana
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

def main():
    # carrega o novo arquivo
    dataset = read_csv('household_power_consumption_days.csv', 
                        header = 0, 
                        infer_datetime_format = True, 
                        parse_dates=['datetime'], 
                        index_col=['datetime'])
    
    # dividir em conjunto de treinamento e teste
    train, test = split_dataset(dataset.values)
    
    # define o número de subsequências e o comprimento das subsequências
    n_steps, n_length = 2, 7
    
    # define o total de dias para usar como entrada
    n_input = n_length * n_steps
    score, scores = evaluate_model(train, test, n_steps, n_length, n_input)
    
    # resumir pontuações
    summarize_scores('lstm', score, scores)
    
    # plotar pontuações
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    pyplot.plot(days, scores, marker='o', label='lstm')
    pyplot.show()
    
    
if __name__ == '__main__':
    main()
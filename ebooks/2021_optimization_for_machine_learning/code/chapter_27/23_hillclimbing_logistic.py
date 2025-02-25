# optimize logistic regression model with a stochastic hill climber
from math import exp
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# logistic regression
def predict_row(row, coefficients):
	# add the bias, the last coefficient
	result = coefficients[-1]
	# add the weighted input
	for i in range(len(row)):
		result += coefficients[i] * row[i]
	# logistic function
	logistic = 1.0 / (1.0 + exp(-result))
	return logistic

# use model coefficients to generate predictions for a dataset of rows
def predict_dataset(X, coefficients):
	yhats = list()
	for row in X:
		# make a prediction
		yhat = predict_row(row, coefficients)
		# store the prediction
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, coefficients):
	# generate predictions for dataset
	yhat = predict_dataset(X, coefficients)
	# round predictions to labels
	yhat = [round(y) for y in yhat]
	# calculate accuracy
	score = accuracy_score(y, yhat)
	return score

# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = solution + randn(len(solution)) * step_size
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %.5f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# define the total iterations
n_iter = 2000
# define the maximum step size
step_size = 0.1
# determine the number of coefficients
n_coef = X.shape[1] + 1
# define the initial solution
solution = rand(n_coef)
# perform the hill climbing search
coefficients, score = hillclimbing(X_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('Coefficients: %s' % coefficients)
print('Train Accuracy: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)
# round predictions to labels
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %f' % (score))

# gradient descent optimization with nadam for a two-dimensional test function
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed

# objective function
def objective(x, y):
	return x**2.0 + y**2.0

# derivative of objective function
def derivative(x, y):
	return asarray([x * 2.0, y * 2.0])

# gradient descent algorithm with nadam
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps=1e-8):
	# generate an initial point
	x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
	# initialize decaying moving averages
	m = [0.0 for _ in range(bounds.shape[0])]
	n = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent
	for t in range(n_iter):
		# calculate gradient g(t)
		g = derivative(x[0], x[1])
		# build a solution one variable at a time
		for i in range(bounds.shape[0]):
			# m(t) = mu * m(t-1) + (1 - mu) * g(t)
			m[i] = mu * m[i] + (1.0 - mu) * g[i]
			# n(t) = nu * n(t-1) + (1 - nu) * g(t)^2
			n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
			# mhat = (mu * m(t) / (1 - mu)) + ((1 - mu) * g(t) / (1 - mu))
			mhat = (mu * m[i] / (1.0 - mu)) + ((1 - mu) * g[i] / (1.0 - mu))
			# nhat = nu * n(t) / (1 - nu)
			nhat = nu * n[i] / (1.0 - nu)
			# x(t) = x(t-1) - alpha / (sqrt(nhat) + eps) * mhat
			x[i] = x[i] - alpha / (sqrt(nhat) + eps) * mhat
		# evaluate candidate point
		score = objective(x[0], x[1])
		# report progress
		print('>%d f(%s) = %.5f' % (t, x, score))
	return [x, score]

# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# steps size
alpha = 0.02
# factor for average gradient
mu = 0.8
# factor for average squared gradient
nu = 0.999
# perform the gradient descent search with nadam
best, score = nadam(objective, derivative, bounds, n_iter, alpha, mu, nu)
print('Done!')
print('f(%s) = %f' % (best, score))

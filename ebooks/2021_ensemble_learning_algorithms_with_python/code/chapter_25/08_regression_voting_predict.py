# make a prediction with a voting ensemble
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
# define the base models
models = list()
for n in range(1,6):
	models.append(('cart'+str(n), DecisionTreeRegressor(max_depth=n)))
# define the voting ensemble
ensemble = VotingRegressor(estimators=models)
# fit the model on all available data
ensemble.fit(X, y)
# make a prediction for one example
row = [0.59332206, -0.56637507, 1.34808718, -0.57054047, -0.72480487, 1.05648449, 0.77744852, 0.07361796, 0.88398267, 2.02843157, 1.01902732, 0.11227799, 0.94218853, 0.26741783, 0.91458143, -0.72759572, 1.08842814, -0.61450942, -0.69387293, 1.69169009]
yhat = ensemble.predict([row])
# summarize prediction
print('Predicted Value: %.3f' % (yhat))
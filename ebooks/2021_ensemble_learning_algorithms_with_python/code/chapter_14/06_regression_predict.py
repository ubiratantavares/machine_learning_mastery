# bagging ensemble for making predictions for regression
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=5)
# define the model
model = BaggingRegressor()
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [0.88950817, -0.93540416, 0.08392824, 0.26438806, -0.52828711, -1.21102238, -0.4499934, 1.47392391, -0.19737726, -0.22252503, 0.02307668, 0.26953276, 0.03572757, -0.51606983, -0.39937452, 1.8121736, -0.00775917, -0.02514283, -0.76089365, 1.58692212]
yhat = model.predict([row])
# summarize the prediction
print('Prediction: %d' % yhat[0])
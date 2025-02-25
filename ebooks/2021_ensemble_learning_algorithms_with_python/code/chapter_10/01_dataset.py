# example of multioutput regression dataset
from sklearn.datasets import make_regression
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
# summarize dataset
print(X.shape, y.shape)
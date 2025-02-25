# synthetic regression dataset
from sklearn.datasets import make_regression
# define dataset
X, y = make_regression(n_samples=1000, n_features=100, noise=0.5, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
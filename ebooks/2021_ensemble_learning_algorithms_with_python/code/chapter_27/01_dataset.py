# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_redundant=10, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
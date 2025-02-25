# multi-class classification dataset
from sklearn.datasets import make_classification
from collections import Counter
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
# summarize the number of examples in each class
print(Counter(y))
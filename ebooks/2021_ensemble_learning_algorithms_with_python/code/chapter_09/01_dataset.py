# multi-class classification dataset
from collections import Counter
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
# summarize the dataset
print(X.shape, y.shape)
# summarize the number of examples in each class
print(Counter(y))
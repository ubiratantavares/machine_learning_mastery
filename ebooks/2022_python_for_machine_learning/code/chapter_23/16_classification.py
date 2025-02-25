from sklearn.datasets import make_classification
from sklearn.svm import SVC
import numpy as np

# Generate 10-dimensional features and 3-class targets
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3,
                           n_informative=4, n_redundant=2, n_repeated=1,
                           random_state=42)

# Run SVC on the data
clf = SVC(kernel="rbf")
clf.fit(X, y)

# Print the accuracy
print(clf.score(X, y))

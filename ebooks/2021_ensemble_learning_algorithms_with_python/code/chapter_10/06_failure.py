# failure of support vector regression for multioutput regression (causes an error)
from sklearn.datasets import make_regression
from sklearn.svm import LinearSVR
# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1)
# define model
model = LinearSVR()
# fit model
# (THIS WILL CAUSE AN ERROR!)
model.fit(X, y)

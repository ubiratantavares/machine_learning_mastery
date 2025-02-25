# make a prediction with logistic regression using one-vs-rest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = LogisticRegression()
# define the one-vs-rest strategy
ovr = OneVsRestClassifier(model)
# fit the model on the whole dataset
ovr.fit(X, y)
# make a single prediction
row = [1.89149379, -0.39847585, 1.63856893, 0.01647165, 1.51892395, -3.52651223, 1.80998823, 0.58810926, -0.02542177, -0.52835426]
yhat = ovr.predict([row])
print('Predicted Class: %d' % yhat[0])
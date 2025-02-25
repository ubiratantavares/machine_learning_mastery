# make a prediction with a soft voting ensemble
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2)
# define the base models
models = list()
for n in range(1,6):
	models.append(('svm'+str(n), SVC(probability=True, kernel='poly', degree=n)))
# define the soft voting ensemble
ensemble = VotingClassifier(estimators=models, voting='soft')
# fit the model on all available data
ensemble.fit(X, y)
# make a prediction for one example
row = [5.88891819, 2.64867662, -0.42728226, -1.24988856, -0.00822, -3.57895574, 2.87938412, -1.55614691, -0.38168784, 7.50285659, -1.16710354, -5.02492712, -0.46196105, -0.64539455, -1.71297469, 0.25987852, -0.193401, -5.52022952, 0.0364453, -1.960039]
yhat = ensemble.predict([row])
# summarize prediction
print('Predicted Class: %d' % (yhat))
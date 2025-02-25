# make predictions using bagging for classification
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
# define the model
model = BaggingClassifier()
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [-4.7705504, -1.88685058, -0.96057964, 2.53850317, -6.5843005, 3.45711663, -7.46225013, 2.01338213, -0.45086384, -1.89314931, -2.90675203, -0.21214568, -0.9623956, 3.93862591, 0.06276375, 0.33964269, 4.0835676, 1.31423977, -2.17983117, 3.1047287]
yhat = model.predict([row])
# summarize the prediction
print('Predicted Class: %d' % yhat[0])
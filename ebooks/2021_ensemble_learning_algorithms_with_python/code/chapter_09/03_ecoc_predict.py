# use error-correcting output codes model as a final model and make a prediction
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OutputCodeClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1, n_classes=3)
# define the binary classification model
model = LogisticRegression()
# define the ecoc model
ecoc = OutputCodeClassifier(model, code_size=2, random_state=1)
# fit the model on the whole dataset
ecoc.fit(X, y)
# make a single prediction
row = [0.04339387, 2.75542632, -3.79522705, -0.71310994, -3.08888853, -1.2963487, -1.92065166, -3.15609907, 1.37532356, 3.61293237, 1.00353523, -3.77126962, 2.26638828, -10.22368666, -0.35137382, 1.84443763, 3.7040748, 2.50964286, 2.18839505, -2.31211692]
yhat = ecoc.predict([row])
print('Predicted Class: %d' % yhat[0])
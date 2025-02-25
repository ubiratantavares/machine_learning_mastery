# evaluate logistic regression for multi-class classification using one-vs-rest
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = LogisticRegression()
# define the one-vs-rest strategy
ovr = OneVsRestClassifier(model)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(ovr, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize the performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
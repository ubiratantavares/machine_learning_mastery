# ensemble of a fixed number of features selected by different feature selection methods
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# get a voting ensemble of models
def get_ensemble(n_features):
	# define the base models
	models = list()
	# anova member
	fs = SelectKBest(score_func=f_classif, k=n_features)
	anova = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('anova', anova))
	# mutual information member
	fs = SelectKBest(score_func=mutual_info_classif, k=n_features)
	mutinfo = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('mutinfo', mutinfo))
	# rfe member
	fs = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features)
	rfe = Pipeline([('fs', fs), ('m', DecisionTreeClassifier())])
	models.append(('rfe', rfe))
	# define the voting ensemble
	ensemble = VotingClassifier(estimators=models, voting='hard')
	return ensemble

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
# get the ensemble model
ensemble = get_ensemble(15)
# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
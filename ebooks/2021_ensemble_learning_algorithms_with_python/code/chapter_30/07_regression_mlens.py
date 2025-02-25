# example of evaluating a super learner ensemble for regression with the mlens library
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from mlens.ensemble import SuperLearner

# create a list of base-models
def get_models():
	models = list()
	models.append(ElasticNet())
	models.append(SVR(gamma='scale'))
	models.append(DecisionTreeRegressor())
	models.append(KNeighborsRegressor())
	models.append(AdaBoostRegressor())
	models.append(BaggingRegressor(n_estimators=10))
	models.append(RandomForestRegressor(n_estimators=10))
	models.append(ExtraTreesRegressor(n_estimators=10))
	return models

# create the inputs and outputs
X, y = make_regression(n_samples=1000, n_features=100, noise=0.5, random_state=1)
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# create the super learner
ensemble = SuperLearner(scorer=mean_absolute_error, folds=10, shuffle=True, sample_size=len(X_train))
# add the base models
ensemble.add(get_models())
# add the meta model
ensemble.add_meta(LinearRegression())
# fit the super learner
ensemble.fit(X_train, y_train)
# summarize base learners
print(ensemble.data)
# make predictions on hold out set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
print('Super Learner MAE: %.3f' % score)
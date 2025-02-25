"""
How to Develop a Framework to Spot-Check Machine Learning Algorithms in Python
https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-python/
"""

import warnings

from numpy import mean, std

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_regression

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, \
    LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor, Lars, LassoLars, \
    PassiveAggressiveRegressor, RANSACRegressor, SGDRegressor, TheilSenRegressor

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor

from sklearn.svm import SVC, SVR

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, AdaBoostRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score


def load_dataset_for_classification(n_samples, n_classes, n_features, n_clusters_per_class, weights, random_state):
    X, y = make_classification(n_samples=n_samples, n_classes=n_classes, n_features=n_features, n_redundant=0,
                               n_clusters_per_class=n_clusters_per_class, weights=weights,
                               flip_y=0, random_state=random_state)
    return X, y


def load_dataset_for_regression(n_samples, n_features, noise, random_state):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    return X, y


# define linear models
def define_linear_models_for_classification():
    models = {'logistic': LogisticRegression()}
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for a in alpha:
        models['ridge-' + str(a)] = RidgeClassifier(alpha=a)
    models['sgd'] = SGDClassifier(max_iter=1000, tol=1e-3)
    models['pa'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
    return models


def define_linear_models_for_regression():
    models = {'lr': LinearRegression()}
    alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for a in alpha:
        models['lasso-' + str(a)] = Lasso(alpha=a)
    for a in alpha:
        models['ridge-' + str(a)] = Ridge(alpha=a)
    for a1 in alpha:
        for a2 in alpha:
            name = 'en-' + str(a1) + '-' + str(a2)
            models[name] = ElasticNet(a1)
    models['huber'] = HuberRegressor()
    models['lars'] = Lars()
    models['llars'] = LassoLars()
    models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
    models['ranscac'] = RANSACRegressor()
    models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
    models['theil'] = TheilSenRegressor()
    return models


# define non-linear models
def define_non_linear_models_for_classification():
    models = {}
    n_neighbors = range(1, 21)
    for k in n_neighbors:
        models['knn-' + str(k)] = KNeighborsClassifier(n_neighbors=k)
    models['cart'] = DecisionTreeClassifier()
    models['extra'] = ExtraTreeClassifier()
    models['svml'] = SVC(kernel='linear')
    models['svmp'] = SVC(kernel='poly')
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for c in c_values:
        models['svmr' + str(c)] = SVC(C=c)
    models['bayes'] = GaussianNB()
    return models


def define_non_linear_models_for_regression():
    models = {}
    # non-linear models
    n_neighbors = range(1, 21)
    for k in n_neighbors:
        models['knn-' + str(k)] = KNeighborsRegressor(n_neighbors=k)
    models['cart'] = DecisionTreeRegressor()
    models['extra'] = ExtraTreeRegressor()
    models['svml'] = SVR(kernel='linear')
    models['svmp'] = SVR(kernel='poly')
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for c in c_values:
        models['svmr' + str(c)] = SVR(C=c)
    return models


# define ensemble models
def define_ensemble_models_for_classification():
    models = {}
    n_trees = 100
    models['ada'] = AdaBoostClassifier(n_estimators=n_trees)
    models['bag'] = BaggingClassifier(n_estimators=n_trees)
    models['rf'] = RandomForestClassifier(n_estimators=n_trees)
    models['et'] = ExtraTreesClassifier(n_estimators=n_trees)
    models['gbm'] = GradientBoostingClassifier(n_estimators=n_trees)
    return models


def define_ensemble_models_for_regression():
    models = {}
    # ensemble models
    n_trees = 100
    models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
    models['bag'] = BaggingRegressor(n_estimators=n_trees)
    models['rf'] = RandomForestRegressor(n_estimators=n_trees)
    models['et'] = ExtraTreesRegressor(n_estimators=n_trees)
    models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
    return models


# no transforms pipeline
def pipeline_none(model):
    return model


# standardize transform pipeline
def pipeline_standardize(model):
    steps = list()
    # standardization
    steps.append(('standardize', StandardScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline


# normalize transform pipeline
def pipeline_normalize(model):
    steps = list()
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline


# standardize and normalize pipeline
def pipeline_std_norm(model):
    steps = list()
    # standardization
    steps.append(('standardize', StandardScaler()))
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline


# evaluate a single model
def evaluate_model(X, y, model, folds, metric, pipeline_func):
    # create the pipeline
    pipeline = pipeline_func(model)
    # evaluate model
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)
    return scores


# evaluate a model and try to trap errors and and hide warnings
def robust_evaluate_model(X, y, model, folds, metric, pipeline_func):
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = evaluate_model(X, y, model, folds, metric, pipeline_func)
    except:
        scores = None
    return scores


# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, y, models, pipe_funcs, folds=10, metric='accuracy'):
    results = dict()
    for name, model in models.items():
        # evaluate model under each preparation function
        for i in range(len(pipe_funcs)):
            # evaluate the model
            scores = robust_evaluate_model(X, y, model, folds, metric, pipe_funcs[i])
            # update name
            run_name = str(i) + name
            # show process
            if scores is not None:
                # store a result
                results[run_name] = scores
                mean_score, std_score = mean(scores), std(scores)
                print('{}: {:.3f} (+/-{:.3f})'.format(name, mean_score, std_score))
            else:
                print('{}: error'.format(name))
    return results


# print and plot the top n results
def summarize_results(results, maximize=True, top_n=10):
    # check for no results
    if len(results) == 0:
        print('no results')
        return
    # determine how many results to summarize
    n = min(top_n, len(results))
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k, mean(v)) for k, v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g. for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
    # retrieve the top n for summarization
    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]
    # print the top n
    print()
    for i in range(n):
        name = names[i]
        mean_score, std_score = mean(results[name]), std(results[name])
        print('Rank={}, Name={}, Score={:.3f} (+/- {:.3f})'.format(i + 1, name, mean_score, std_score))
    # boxplot for the top n
    plt.boxplot(scores, labels=names)
    _, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.grid()
    plt.show()
    plt.savefig('spotcheck.png')


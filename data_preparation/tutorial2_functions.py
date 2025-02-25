"""
Linear Discriminant Analysis for Dimensionality Reduction in Python
https://machinelearningmastery.com/linear-discriminant-analysis-for-dimensionality-reduction-in-python/
"""

# evaluate lda with naive bayes algorithm for classification
from numpy import mean, std, array, min
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt


# get the dataset
def get_dataset(n_samples, n_features, n_informative, n_redundant, random_state, n_classes):
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=n_redundant,
                               random_state=random_state,
                               n_classes=n_classes)
    return X, y


# get a list of models to evaluate
def get_models(n_features,  n_classes):
    lista = array([n_features,  n_classes])
    minimo = min(lista)
    print(minimo)
    models = dict()
    for i in range(1, minimo):
        steps = [('lda', LinearDiscriminantAnalysis(n_components=i)), ('m', GaussianNB())]
        models[str(i)] = Pipeline(steps=steps)
    return models


def get_model():
    steps = [('lda', LinearDiscriminantAnalysis(n_components=9)), ('m', GaussianNB())]
    model = Pipeline(steps=steps)
    return model


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# evaluate the models and store results
def evaluate_models_and_store_results(models, X, y):
    names, results = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        names.append(name)
        results.append(scores)
        print('>{} Accuracy: {:.3f} ({:.3f})'.format(name, mean(scores), std(scores)))
    return names, results


# plot model performance for comparison
def plot_models(names, results):
    plt.boxplot(results, labels=names, showmeans=True)
    plt.grid()
    plt.show()

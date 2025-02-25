"""
6 Dimensionality Reduction Algorithms With Python
https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/
"""

from numpy import mean, std

from sklearn.datasets import make_classification

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold


def load_dataset_for_classification(n_samples, n_classes, n_features, n_clusters_per_class, weights, random_state):
    X, y = make_classification(n_samples=n_samples,
                               n_classes=n_classes,
                               n_features=n_features,
                               n_informative=int(n_features/2),
                               n_redundant=int(n_features/2),
                               n_clusters_per_class=n_clusters_per_class,
                               weights=weights,
                               flip_y=0,
                               random_state=random_state)
    return X, y


def define_evaluation_procedure(n_splits=10, n_repeats=3, random_state=1):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return cv


# no transforms pipeline
def pipeline_none(model):
    return model


# Principal Component Analysis
def pipeline_pca(model, n_components=10):
    steps = [('pca', PCA(n_components=n_components)), ('model', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline


# Singular Value Decomposition
def pipeline_svd(model, n_components=10):
    steps = [('svd', TruncatedSVD(n_components=n_components)), ('model', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline


# Linear Discriminant Analysis
def pipeline_lda(model, n_components=1):
    steps = [('lda', LinearDiscriminantAnalysis(n_components=n_components)), ('model', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline


# Isomap Embedding
def pipeline_iso(model, n_components=10):
    steps = [('iso', Isomap(n_components=n_components)), ('model', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline


# Locally Linear Embedding
def pipeline_lle(model, n_components=10):
    steps = [('lle', LocallyLinearEmbedding(n_components=n_components)), ('model', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline


# Modified Locally Linear Embedding
def pipeline_lle_m(model, n_components=5):
    steps = [('lle_m', LocallyLinearEmbedding(n_components=n_components, method='modified', n_neighbors=10)), ('model', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline


# evaluate a single model
def evaluate_model(X, y, model, metric, cv, pipeline_func):
    # create the pipeline
    pipeline = pipeline_func(model)
    # evaluate model
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=cv, n_jobs=-1, verbose=0)
    return scores


def summarize_performance(model, scores, scoring):
    print('{} => {}: average={:.3f}, standard deviation:{:.3f}'.
          format(model.__str__(), scoring, mean(scores), std(scores)))

"""
Cost-Sensitive SVM for Imbalanced Classification
https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/
"""


from imbalanced_classification import functions as f
from sklearn.svm import SVC


# define dataset
# define_dataset(n_samples, n_features, n_clusters_per_class, weights, random_state)
X, y = f.define_dataset(10_000, 2, 1, [0.99], 4)

# summarize class distribution
f.summarize_class_distribution(y)

# scatter plot of examples by class label
f.scatter_plot_by_class_label(X, y)

# define evaluation procedure
cv = f.define_evaluation_procedure()

# configuração dos pesos: weights = {0: 1.0, 1: 100.0}, weights = {0: 0.01, 1: 1.0}, weights = "balanced"
weight = "balanced"

# define models
models = [SVC(gamma='scale'), SVC(gamma='scale', class_weight=weight)]

for model in models:
    # evaluate model
    scores = f.evaluate_model(model, X, y, cv)
    # summarize performance
    f.summarize_performance(model, scores)

"""
Grid Search
"""

weights = [{0: 1, 1: 1},
           {0: 1, 1: 10},
           {0: 1, 1: 100},
           {0: 1, 1: 1000},
           {0: 0.1, 1: 1},
           {0: 0.01, 1: 1},
           {0:0.001, 1: 1}]

param_grid = dict(class_weight=weights)

# define grid search
grid_search = f.define_grid_search(models[0], param_grid, cv)

# execute the grid search
grid_result = grid_search.fit(X, y)

# report the best configuration
f.report_the_best_configuration(models[0], grid_result)

# report all configurations
f.report_all_configurations(models[0], grid_result)

"""
Counter({0: 9900, 1: 100})
SVC() => ROC AUC: average=0.808, standard deviation:0.126
SVC(class_weight='balanced') => ROC AUC: average=0.967, standard deviation:0.039
SVC() => Best: 0.975 using {'class_weight': {0: 0.01, 1: 1}}
SVC() => 0.808 (0.126) with: {'class_weight': {0: 1, 1: 1}}
SVC() => 0.934 (0.065) with: {'class_weight': {0: 1, 1: 10}}
SVC() => 0.968 (0.039) with: {'class_weight': {0: 1, 1: 100}}
SVC() => 0.936 (0.038) with: {'class_weight': {0: 1, 1: 1000}}
SVC() => 0.900 (0.083) with: {'class_weight': {0: 0.1, 1: 1}}
SVC() => 0.975 (0.028) with: {'class_weight': {0: 0.01, 1: 1}}
SVC() => 0.957 (0.026) with: {'class_weight': {0: 0.001, 1: 1}}
"""
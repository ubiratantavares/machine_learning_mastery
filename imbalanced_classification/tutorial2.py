"""
Cost-Sensitive Decision Trees for Imbalanced Classification
https://machinelearningmastery.com/cost-sensitive-decision-trees-for-imbalanced-classification/

"""

from imbalanced_classification import functions as f
from sklearn.tree import DecisionTreeClassifier

# define dataset
# define_dataset(n_samples, n_features, n_clusters_per_class, weights, random_state)
X, y = f.define_dataset(10_000, 2, 1, [0.99], 3)

# summarize class distribution
f.summarize_class_distribution(y)

# scatter plot of examples by class label
f.scatter_plot_by_class_label(X, y)

# define evaluation procedure
cv = f.define_evaluation_procedure()

# configuração dos pesos: weights = {0: 1.0, 1: 100.0}, weights = {0: .01, 1: 1.0}, weights = "balanced"
weight = "balanced"

# define model - sem ponderação de classe
models = [DecisionTreeClassifier(), DecisionTreeClassifier(class_weight=weight)]

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
           {0: 0.001, 1: 1}]

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
DecisionTreeClassifier() => ROC AUC: average=0.838, standard deviation:0.081
DecisionTreeClassifier(class_weight='balanced') => ROC AUC: average=0.828, standard deviation:0.075
DecisionTreeClassifier() => Best: 0.838 using {'class_weight': {0: 1, 1: 1}}
DecisionTreeClassifier() => 0.838 (0.081) with: {'class_weight': {0: 1, 1: 1}}
DecisionTreeClassifier() => 0.833 (0.075) with: {'class_weight': {0: 1, 1: 10}}
DecisionTreeClassifier() => 0.827 (0.076) with: {'class_weight': {0: 1, 1: 100}}
DecisionTreeClassifier() => 0.829 (0.075) with: {'class_weight': {0: 1, 1: 1000}}
DecisionTreeClassifier() => 0.833 (0.075) with: {'class_weight': {0: 0.1, 1: 1}}
DecisionTreeClassifier() => 0.828 (0.075) with: {'class_weight': {0: 0.01, 1: 1}}
DecisionTreeClassifier() => 0.829 (0.075) with: {'class_weight': {0: 0.001, 1: 1}}
"""
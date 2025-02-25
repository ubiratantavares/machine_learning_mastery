"""
Cost-Sensitive Logistic Regression for Imbalanced Classification
https://machinelearningmastery.com/cost-sensitive-logistic-regression/
"""


from imbalanced_classification import functions as f
from sklearn.linear_model import LogisticRegression

# define dataset
# define_dataset(n_samples, n_features, n_clusters_per_class, weights, random_state)
X, y = f.define_dataset(10_000, 2, 1, [0.99], 2)

# summarize class distribution
f.summarize_class_distribution(y)

# scatter plot of examples by class label
f.scatter_plot_by_class_label(X, y)

# define evaluation procedure
cv = f.define_evaluation_procedure()

# configuração dos pesos: weights = {0: 1.0, 1: 100.0}, weights = {0: 0.01, 1: 1.0}, weights = "balanced"
weight = "balanced"

# define models
models = [LogisticRegression(solver='lbfgs'), LogisticRegression(solver='lbfgs', class_weight=weight)]

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
           {0: .1, 1: 1},
           {0: .01, 1: 1},
           {0: .001, 1: 1}]

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
LogisticRegression() => ROC AUC: average=0.974, standard deviation:0.034
LogisticRegression(class_weight='balanced') => ROC AUC: average=0.978, standard deviation:0.027
LogisticRegression() => Best: 0.978 using {'class_weight': {0: 0.01, 1: 1}}
LogisticRegression() => 0.974 (0.034) with: {'class_weight': {0: 1, 1: 1}}
LogisticRegression() => 0.975 (0.033) with: {'class_weight': {0: 1, 1: 10}}
LogisticRegression() => 0.978 (0.027) with: {'class_weight': {0: 1, 1: 100}}
LogisticRegression() => 0.973 (0.023) with: {'class_weight': {0: 1, 1: 1000}}
LogisticRegression() => 0.977 (0.030) with: {'class_weight': {0: 0.1, 1: 1}}
LogisticRegression() => 0.978 (0.025) with: {'class_weight': {0: 0.01, 1: 1}}
LogisticRegression() => 0.977 (0.022) with: {'class_weight': {0: 0.001, 1: 1}}
"""
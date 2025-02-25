"""
Bagging and Random Forest for Imbalanced Classification
https://machinelearningmastery.com/bagging-and-random-forest-for-imbalanced-classification/
"""


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier
from imbalanced_classification import functions as f


# define dataset
# define_dataset(n_samples, n_features, n_clusters_per_class, weights, random_state)
X, y = f.define_dataset(10_000, 2, 1, [0.99], 4)

# define evaluation procedure
cv = f.define_evaluation_procedure()

# define models
models = [BaggingClassifier(),
          BalancedBaggingClassifier(),
          RandomForestClassifier(n_estimators=10),
          RandomForestClassifier(n_estimators=10, class_weight='balanced'),
          RandomForestClassifier(n_estimators=10, class_weight='balanced_subsample'),
          BalancedRandomForestClassifier(n_estimators=10),
          EasyEnsembleClassifier(n_estimators=10)]


for model in models:
    # evaluate model
    scores = f.evaluate_model(model, X, y, cv)
    # summarize performance
    f.summarize_performance(model, scores)


weights = [{0: 1, 1: 1},
           {0: 1, 1: 10},
           {0: 1, 1: 100},
           {0: 1, 1: 1000},
           {0: .1, 1: 1},
           {0: .01, 1: 1},
           {0: .001, 1: 1}]

param_grid = dict(class_weight=weights)

# define grid search
grid_search = f.define_grid_search(models[2], param_grid, cv)

# execute the grid search
grid_result = grid_search.fit(X, y)

# report the best configuration
f.report_the_best_configuration(models[2], grid_result)

# report all configurations
f.report_all_configurations(models[2], grid_result)

"""
BaggingClassifier() => ROC AUC: average=0.869, standard deviation:0.079
BalancedBaggingClassifier() => ROC AUC: average=0.959, standard deviation:0.041
RandomForestClassifier(n_estimators=10) => ROC AUC: average=0.864, standard deviation:0.075
RandomForestClassifier(class_weight='balanced', n_estimators=10) => ROC AUC: average=0.871, standard deviation:0.077
RandomForestClassifier(class_weight='balanced_subsample', n_estimators=10) => ROC AUC: average=0.879, standard deviation:0.071
BalancedRandomForestClassifier(n_estimators=10) => ROC AUC: average=0.967, standard deviation:0.035
EasyEnsembleClassifier() => ROC AUC: average=0.963, standard deviation:0.043
RandomForestClassifier(n_estimators=10) => Best: 0.881 using {'class_weight': {0: 0.1, 1: 1}}
RandomForestClassifier(n_estimators=10) => 0.874 (0.078) with: {'class_weight': {0: 1, 1: 1}}
RandomForestClassifier(n_estimators=10) => 0.877 (0.077) with: {'class_weight': {0: 1, 1: 10}}
RandomForestClassifier(n_estimators=10) => 0.872 (0.076) with: {'class_weight': {0: 1, 1: 100}}
RandomForestClassifier(n_estimators=10) => 0.877 (0.073) with: {'class_weight': {0: 1, 1: 1000}}
RandomForestClassifier(n_estimators=10) => 0.881 (0.074) with: {'class_weight': {0: 0.1, 1: 1}}
RandomForestClassifier(n_estimators=10) => 0.878 (0.078) with: {'class_weight': {0: 0.01, 1: 1}}
RandomForestClassifier(n_estimators=10) => 0.876 (0.070) with: {'class_weight': {0: 0.001, 1: 1}}
"""
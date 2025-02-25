"""
How to Develop a Framework to Spot-Check Machine Learning Algorithms in Python
https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-python/
"""

from python_machine_learning import functions as f

"""
binary classification spot check script
"""

# load dataset
X, y = f.load_dataset_for_classification(n_samples=1000, n_classes=2, n_features=2, n_clusters_per_class=1,
                                         weights=None, random_state=3)

# get model list
models = f.define_linear_models_for_classification()
models2 = f.define_non_linear_models_for_classification()
models3 = f.define_ensemble_models_for_classification()
models.update(models2)
models.update(models3)

print(len(models))

# define transform pipelines
pipelines = [f.pipeline_none, f.pipeline_standardize, f.pipeline_normalize, f.pipeline_std_norm]

# evaluate models
results = f.evaluate_models(X, y, models, pipelines)

# summarize results
f.summarize_results(results, top_n=3)


"""
regression spot check script

X, y = f.load_dataset_for_regression(1000, 50, 0.1, 1)

# get model list
models = f.define_linear_models_for_regression()
models2 = f.define_non_linear_models_for_regression()
models3 = f.define_ensemble_models_for_regression()
models.update(models2)
models.update(models3)


# evaluate models
results = f.evaluate_models(X, y, models, pipelines, metric='neg_mean_squared_error')

# summarize results
f.summarize_results(results)

"""




"""
6 Dimensionality Reduction Algorithms With Python
https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/
"""


from data_preparation import tutorial1_functions as tf
from sklearn.linear_model import LogisticRegression

# define dataset
# n_samples, n_classes, n_features, n_clusters_per_class, weights, random_state
X, y = tf.load_dataset_for_classification(1_000, 2, 20, 1, None, 7)
print(X.shape, y.shape)

# define models
models = [LogisticRegression()]

# define pipelines
pipelines = [tf.pipeline_none, tf.pipeline_pca, tf.pipeline_svd, tf.pipeline_lda,
             tf.pipeline_iso, tf.pipeline_lle, tf.pipeline_lle_m]

# evaluate model
cv = tf.define_evaluation_procedure()

metric = 'accuracy'

for model in models:
    for pipeline in pipelines:
        scores = tf.evaluate_model(X, y, model, metric, cv, pipeline)
        tf.summarize_performance(model, scores, metric)






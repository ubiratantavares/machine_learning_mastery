"""
Linear Discriminant Analysis for Dimensionality Reduction in Python
https://machinelearningmastery.com/linear-discriminant-analysis-for-dimensionality-reduction-in-python/
"""


from data_preparation import tutorial2_functions as tf


# define dataset
n_samples, n_features, n_informative, n_redundant, random_state, n_classes = 1000, 20, 15, 5, 7, 10
X, y = tf.get_dataset(n_samples, n_features, n_informative, n_redundant, random_state, n_classes)
print(X.shape, y.shape)

# get the models to evaluate
models = tf.get_models(n_features,  n_classes)

names, results = tf.evaluate_models_and_store_results(models, X, y)

tf.plot_models(names, results)

model = tf.get_model()

# fit the model on the whole dataset
model.fit(X, y)

# make a single prediction
row = [[2.3548775, -1.69674567, 1.6193882, -1.19668862, -2.85422348, -2.00998376, 16.56128782, 2.57257575,
        9.93779782, 0.43415008, 6.08274911, 2.12689336, 1.70100279, 3.32160983, 13.02048541, -3.05034488,
        2.06346747, -3.33390362, 2.45147541, -1.23455205]]

yhat = model.predict(row)

print('Predicted Class: {}'.format(yhat[0]))

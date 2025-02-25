"""
Singular Value Decomposition for Dimensionality Reduction in Python
https://machinelearningmastery.com/singular-value-decomposition-for-dimensionality-reduction-in-python/
"""

from data_preparation import tutorial3_functions as tf


# define dataset
n_samples, n_features, n_informative, n_redundant, random_state, n_classes = 1000, 20, 15, 5, 7, 2
X, y = tf.get_dataset(n_samples, n_features, n_informative, n_redundant, random_state, n_classes)
print(X.shape, y.shape)

# get the models to evaluate
models = tf.get_models(n_features,  n_classes)

# evaluate the models and store results
names, results = tf.evaluate_models_and_store_results(models, X, y)

tf.plot_models(names, results)

model = tf.get_model(15)

# fit the model on the whole dataset
model.fit(X, y)

# make a single prediction
row = [[0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719, 0.28422388, -7.1827928,
        -1.91211104, 2.73729512, 0.81395695,3.96973717, -2.66939799, 3.34692332, 4.19791821,
        0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]]

yhat = model.predict(row)

print('Predicted Class: {}'.format(yhat[0]))

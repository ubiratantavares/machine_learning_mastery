"""
How to Develop a Cost-Sensitive Neural Network for Imbalanced Classification
https://machinelearningmastery.com/cost-sensitive-neural-network-for-imbalanced-classification/
"""

from imbalanced_classification import functions as f

# define dataset
X, y = f.define_dataset(10_000, 2, 2, [0.99], 4)

# summarize class distribution
f.summarize_class_distribution(y)

# scatter plot of examples by class label
f.scatter_plot_by_class_label(X, y)

# prepare train and test dataset
trainX, trainy, testX, testy = f.prepare_train_and_test(X, y, 5000)

# define the model
n_input = trainX.shape[1]

# define the neural network model
model = f.define_model_rna(n_input)

# fit model
model.fit(trainX, trainy, epochs=100, verbose=0)

# make predictions on the test dataset
yhat = model.predict(testX)

# evaluate the ROC AUC of the predictions
f.evaluate_ROC_AUC(testy, yhat)
print('\n')

weights = [{0: 1, 1: 1},
           {0: 1, 1: 10},
           {0: 1, 1: 100},
           {0: 1, 1: 1000},
           {0: .1, 1: 1},
           {0: .01, 1: 1},
           {0: .001, 1: 1}]

for weight in weights:

    history = model.fit(trainX, trainy, class_weight=weight, epochs=100, verbose=0)

    # evaluate model
    yhat = model.predict(testX)

    # evaluate the ROC AUC of the predictions
    f.evaluate_ROC_AUC(testy, yhat)

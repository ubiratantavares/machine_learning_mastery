"""
Naive Bayes Classifier From Scratch in Python
https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

from code_algorithms_from_scratch.tutorial1_functions import *

dataset = dataset_test()
print(dataset[0])

print('\n')

separated = separate_by_class(dataset)

for label in separated:
    print(label)
    for row in separated[label]:
        print(row)

print('\n')

summary = summarize_dataset(dataset)

print(summary)

print('\n')

summaries = summarize_dataset_by_class(dataset)

print(summaries)

print('\n')

for label in summaries:
    print(label)

    for row in summary[label]:
        print(row)

print('\n')

# Test Gaussian PDF
print(calculate_probability(1.0, 1.0, 1.0))
print(calculate_probability(0.0, 1.0, 1.0))
print(calculate_probability(2.0, 1.0, 1.0))

print('\n')

print(calculate_probability(1.0, 1.0, .5))
print(calculate_probability(0.0, 1.0, .5))
print(calculate_probability(2.0, 1.0, .5))

print('\n')

for i in range(len(dataset)):
    probabilities = calculate_class_probabilities(summaries, dataset[i])
    print(probabilities)

print('\n')

# Make a prediction with Naive Bayes on Iris Dataset
filename = 'iris.csv'

dataset = load_csv(filename)

for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)


# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)

# fit model
model = summarize_dataset_by_class(dataset)

# define a new record
row = [5.7, 2.9, 4.2, 1.3]

# predict the label
label = predict(model, row)

print('\nData={}, Predicted: {}'.format(row, label))


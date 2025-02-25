"""
Naive Bayes Classifier From Scratch in Python
https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

from math import sqrt, pi, exp
from csv import reader

"""
Part I - Naive Bayes Tutorial (in 5 easy steps)
"""

"""
Step 1: Separate By Class
"""


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated


"""
Step 2: Summarize Dataset
"""


# Calculate the mean of a list of numbers
def _mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def _stdev(numbers):
    avg = _mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(_mean(column), _stdev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


"""
Step 3: Summarize Data By Class
"""


# Split dataset by class then calculate statistics for each row
def summarize_dataset_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


"""
Step 4: Gaussian Probability Density Function
"""


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-0.5 * ((x - mean) ** 2 / (stdev ** 2)))
    return (1 / (stdev * sqrt(2 * pi))) * exponent


# Naive Bayes
# P(classe=0 | X1,X2) = P(X1|classe=0) * P(X2|classe=0) * P(classe=0)
# P(classe=1 | X1,X2) = P(X1|classe=1) * P(X2|classe=1) * P(classe=1)

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


# Test summarizing a dataset
def dataset_test():
    lst = [[3.393533211, 2.331273381, 0],
           [3.110073483, 1.781539638, 0],
           [1.343808831, 3.368360954, 0],
           [3.582294042, 4.67917911, 0],
           [2.280362439, 2.866990263, 0],
           [7.423436942, 4.696522875, 1],
           [5.745051997, 3.533989803, 1],
           [9.172168622, 2.511101045, 1],
           [7.792783481, 3.424088941, 1],
           [7.939820817, 0.791637231, 1]]
    return lst


"""
Part II - Iris Flower Species Case Study
"""

# Make Predictions with Naive Bayes On The Iris Dataset


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

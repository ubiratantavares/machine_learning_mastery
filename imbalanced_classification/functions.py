from numpy import where, mean, std

from collections import Counter

from matplotlib import pyplot as plt

from pandas import read_csv
from pandas.plotting import scatter_matrix

from sklearn.datasets import make_classification

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

from keras.layers import Dense
from keras.models import Sequential


def define_dataset(n_samples, n_features, n_clusters_per_class, weights, random_state):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=0,
                               n_clusters_per_class=n_clusters_per_class, weights=weights,
                               flip_y=0, random_state=random_state)
    return X, y


# load the csv file as a data frame
def load_csv(filename, header=None, sep=None):
    df = read_csv(filename, header=header, sep=sep, engine='python')
    return df

# split into input and output elements
def split_input_output_data(df):
    # retrieve numpy array
    data = df.values
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X, y

def define_evaluation_procedure(n_splits=10, n_repeats=3, random_state=1):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return cv

def define_grid_search(model, param_grid, cv):
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=-1, cv=cv, verbose=0)
    return grid

def evaluate_model(model, X, y, cv):
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=0)
    return scores

def summarize_performance(scores):
    return mean(scores), std(scores)

def print_summarize_performance(name_model, scores):
    mean_scores, std_scores = summarize_performance(scores)
    print('{}: {:.3f} +/- {:.3f}'.format(name_model, mean_scores, std_scores))

def summarize_class_distribution(y):
    counter = Counter(y)
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class= {}, Count= {}, Percentage= {:.2f}%'.format(k, v, per))

# summarize the shape of the dataset
def summarize_data(dataframe):
    print(dataframe.shape)

def report_the_best_configuration(model_name, grid_result):
    print("{} => Best: {:.3f} using {}".format(model_name, grid_result.best_score_, grid_result.best_params_))

def report_all_configurations(model_name, grid_result):
    means, stds, params = grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], \
                          grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("{} => {:.3f} ({:.3f}) with: {}".format(model_name, mean, stdev, param))

def scatter_plot_by_class_label(X, y):
    counter = Counter(y)
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    plt.legend()
    plt.grid()
    plt.show()

# prepare train and test dataset
def prepare_train_and_test(X, y, n_train):
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy

def define_model_rna(n_input):
	# define model
	model = Sequential()
	# define first hidden layer and visible layer
	model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
	# define output layer
	model.add(Dense(1, activation='sigmoid'))
	# define loss and optimizer
	model.compile(loss='binary_crossentropy', optimizer='sgd')
	return model

def evaluate_ROC_AUC(testy, yhat):
    score = roc_auc_score(testy, yhat)
    print('ROC AUC: {:.3f}'.format(score))

# histograms of all variables
def histograms_of_all_variables(df):
    df.hist()
    plt.show()

def scatter_plot_matrix(df, color_dict):
    # map each row to a color based on the class value
    colors = [color_dict[str(x)] for x in df.values[:, -1]]
    # pairwise scatter plots of all numerical variables
    scatter_matrix(df, diagonal='kde', color=colors)
    plt.show()

# define models to test
def get_models():
    models_dict = {'LR': LogisticRegression(solver='lbfgs'),
                   'SVM': SVC(gamma='scale'),
                   'BAG': BaggingClassifier(n_estimators=1000),
                   'RF': RandomForestClassifier(n_estimators=1000),
                   'GBM': GradientBoostingClassifier(n_estimators=1000)}
    return models_dict


# define models to test
def get_models2():
    models_dict = {'LR': LogisticRegression(solver='lbfgs', class_weight='balanced'),
                   'SVM': SVC(gamma='scale', class_weight='balanced'),
                   'RF': RandomForestClassifier(n_estimators=1000, class_weight='balanced')}
    return models_dict

# define pipeline steps
def defines_pipeline_steps(name, model):
    steps = [('p', PowerTransformer()), (name, model)]
    # define pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline

# evaluate each model no pipeline
def evaluate_each_model_no_pipeline(models_dict, X, y, cv):
    results_scores = list()
    for name, model in models_dict.items():
        # evaluate the model and store results
        scores = evaluate_model(model, X, y, cv)
        # summarize performance
        mean_scores, std_scores = summarize_performance(scores)
        dic = {'name': name, 'model': model, 'mean': mean_scores, 'std': std_scores, 'scores': scores}
        results_scores.append(dic)
    results_scores_order = sorted(results_scores, key=lambda dicionario: dicionario['mean'], reverse=True)
    return results_scores_order

# evaluate each model with pipeline
def evaluate_each_model_with_pipeline(models_dict, X, y, cv):
    results_scores = list()
    for name, model in models_dict.items():
        pipeline = defines_pipeline_steps(name, model)
        # evaluate the model and store results
        scores = evaluate_model(pipeline, X, y, cv)
        # summarize performance
        mean_scores, std_scores = summarize_performance(scores)
        dic = {'name': name, 'pipeline': pipeline, 'mean': mean_scores, 'std': std_scores, 'scores': scores}
        results_scores.append(dic)
    results_scores_order = sorted(results_scores, key=lambda dicionario: dicionario['mean'], reverse=True)
    return results_scores_order

# plot the results
def plot_results(results_scores_order):
    names = []
    scores = []
    for result in results_scores_order:
        scores.append(result['scores'])
        names.append(result['name'])
    plt.boxplot(scores, labels=names, showmeans=True)
    plt.grid()
    plt.show()

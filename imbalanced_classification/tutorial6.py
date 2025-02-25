"""
Imbalanced Classification Model to Detect Mammography Microcalcifications
https://machinelearningmastery.com/imbalanced-classification-model-to-detect-microcalcifications/
"""
from imbalanced_classification.functions import *
from sklearn.dummy import DummyClassifier

# load the csv file as a data frame
df = load_csv("../dataset/mammography.csv")

y = df.values[:, -1]

# summarize the shape of the dataset
summarize_data(df)

# summarize the class distribution
summarize_class_distribution(y)

# histograms of all variables
histograms_of_all_variables(df)

# define a mapping of class values to colors
color_dict = {"'-1'": 'blue', "'1'": 'red'}

scatter_plot_matrix(df, color_dict)

# split into input and output elements
X, y = split_input_output_data(df)

# summarize the loaded dataset
print(X.shape, y.shape)

# summarize the class distribution
summarize_class_distribution(y)

# define the reference model
model = {'DC': DummyClassifier(strategy='stratified')}

cv = define_evaluation_procedure()

# evaluate the model
scores = evaluate_model(model['DC'], X, y, cv)

print_summarize_performance('DC', scores)

# define models to test
# models_dict = get_models()

# results = evaluate_each_model_no_pipeline(models_dict, X, y, cv)

# plot the results
# plot_results(results)

models_dict = get_models2()

results_scores_order = evaluate_each_model_with_pipeline(models_dict, X, y, cv)

for result in results_scores_order:
    print_summarize_performance(result['name'], result['scores'])

# plot the results
plot_results(results_scores_order)

# power transform then fit model - modelo selecionado
pipeline = results_scores_order[0]['pipeline']

# fit the model
pipeline.fit(X, y)

datas = [[0.23001961, 5.0725783, -0.27606055, 0.83244412, -0.37786573, 0.4803223],
         [0.15549112, -0.16939038, 0.67065219, -0.85955255, -0.37786573, -0.94572324],
         [-0.78441482, -0.44365372, 5.6747053, -0.85955255, -0.37786573, -0.94572324],
         [2.0158239, 0.15353258, -0.32114211, 2.1923706, -0.37786573, 0.96176503],
         [2.3191888, 0.72860087, -0.50146835, -0.85955255, -0.37786573, -0.94572324],
         [0.19224721, -0.2003556, -0.230979, 1.2003796, 2.2620867, 1.132403]]

for i in range(len(datas)):
    # make prediction
    yhat = pipeline.predict([datas[i]])
    # get the label
    label = yhat[0]
    # summarize
    if i == 0 or i == 1 or i == 2:
        print('Predicted={} (expected 0)'.format(label))
    else:
        print('Predicted={} (expected 1)'.format(label))

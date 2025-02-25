
By [Vinod Chugani](https://machinelearningmastery.com/author/vbpm1401/ "Posts by Vinod Chugani") on September 4, 2024 in [Data Science](https://machinelearningmastery.com/category/data-science/ "View all items in Data Science") [0](https://machinelearningmastery.com/filling-the-gaps-a-comparative-guide-to-imputation-techniques-in-machine-learning/#respond)

In our previous exploration of penalized regression models such as Lasso, Ridge, and ElasticNet, we demonstrated how effectively these models manage multicollinearity, allowing us to utilize a broader array of features to enhance model performance. Building on this foundation, we now address another crucial aspect of data preprocessing—handling missing values. Missing data can significantly compromise the accuracy and reliability of models if not appropriately managed. This post explores various imputation strategies to address missing data and embed them into our pipeline. This approach allows us to further refine our predictive accuracy by incorporating previously excluded features, thus making the most of our rich dataset.

## Overview

This post is divided into three parts; they are:

- Reconstructing Manual Imputation with SimpleImputer
- Advancing Imputation Techniques with IterativeImputer
- Leveraging Neighborhood Insights with KNN Imputation

## Reconstructing Manual Imputation with SimpleImputer

In part one of this post, we revisit and reconstruct our earlier manual imputation techniques using `SimpleImputer`. Our previous exploration of the Ames Housing dataset provided foundational insights into [using the data dictionary](https://machinelearningmastery.com/classifying_variables/) to tackle missing data. We demonstrated manual imputation strategies tailored to different data types, considering domain knowledge and data dictionary details. For example, categorical variables missing in the dataset often indicate an absence of the feature (e.g., a missing ‘PoolQC’ might mean no pool exists), guiding our imputation to fill these with “None” to preserve the dataset’s integrity. Meanwhile, numerical features were handled differently, employing techniques like mean imputation.

Now, by automating these processes with scikit-learn’s `SimpleImputer`, we enhance reproducibility and efficiency. Our pipeline approach not only incorporates imputation but also scales and encodes features, preparing them for regression analysis with models such as Lasso, Ridge, and ElasticNet:

```Python
# Import the necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Exclude 'PID' and 'SalePrice' from features and specifically handle the 'Electrical' column
numeric_features = Ames.select_dtypes(include=['int64', 'float64']).drop(columns=['PID', 'SalePrice']).columns
categorical_features = Ames.select_dtypes(include=['object']).columns.difference(['Electrical'])
electrical_feature = ['Electrical']  # Specifically handle the 'Electrical' column

# Helper function to fill 'None' for missing categorical data
def fill_none(X):
    return X.fillna("None")

# Pipeline for numeric features: Impute missing values then scale
numeric_transformer = Pipeline(steps=[
    ('impute_mean', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline for general categorical features: Fill missing values with 'None' then apply one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('fill_none', FunctionTransformer(fill_none, validate=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Specific transformer for 'Electrical' using the mode for imputation
electrical_transformer = Pipeline(steps=[
    ('impute_electrical', SimpleImputer(strategy='most_frequent')),
    ('onehot_electrical', OneHotEncoder(handle_unknown='ignore'))
])

# Combined preprocessor for numeric, general categorical, and electrical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('electrical', electrical_transformer, electrical_feature)
    ])

# Target variable
y = Ames['SalePrice']

# All features
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]

# Define the model pipelines with preprocessor and regressor
models = {
    'Lasso': Lasso(max_iter=20000),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet()
}

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y)
    results[name] = round(scores.mean(), 4)

# Output the cross-validation scores
print("Cross-validation scores with Simple Imputer:", results)
```


The results from this implementation are displayed, showing how simple imputation affects model accuracy and establishes a benchmark for more sophisticated methods discussed later:

```Bash
Cross-validation scores with Simple Imputer: {'Lasso': 0.9138, 'Ridge': 0.9134, 'ElasticNet': 0.8752}
```



Transitioning from manual steps to a pipeline approach using scikit-learn enhances several aspects of data processing:

1. **Efficiency and Error Reduction:** Manually imputing values is time-consuming and prone to errors, especially as data complexity increases. The pipeline automates these steps, ensuring consistent transformations and reducing mistakes.
2. **Reusability and Integration:** Manual methods are less reusable. In contrast, pipelines encapsulate the entire preprocessing and modeling steps, making them easily reusable and seamlessly integrated into the model training process.
3. **Data Leakage Prevention:** There’s a risk of data leakage with manual imputation, as it may include test data when computing values. Pipelines prevent this risk with the fit/transform methodology, ensuring calculations are derived only from the training set.

This framework, demonstrated with `SimpleImputer`, shows a flexible approach to data preprocessing that can be easily adapted to include various imputation strategies. In upcoming sections, we will explore additional techniques, assessing their impact on model performance.
## Advancing Imputation Techniques with IterativeImputer

In part two, we experiment with `IterativeImputer`, a more advanced imputation technique that models each feature with missing values as a function of other features in a round-robin fashion. Unlike simple methods that might use a general statistic such as the mean or median, Iterative Imputer models each feature with missing values as a dependent variable in a regression, informed by the other features in the dataset. This process iterates, refining estimates for missing values using the entire set of available feature interactions. This approach can unveil subtle data patterns and dependencies not captured by simpler imputation methods:

```Python
# Import the necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # This line is needed for IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Exclude 'PID' and 'SalePrice' from features and specifically handle the 'Electrical' column
numeric_features = Ames.select_dtypes(include=['int64', 'float64']).drop(columns=['PID', 'SalePrice']).columns
categorical_features = Ames.select_dtypes(include=['object']).columns.difference(['Electrical'])
electrical_feature = ['Electrical']  # Specifically handle the 'Electrical' column

# Helper function to fill 'None' for missing categorical data
def fill_none(X):
    return X.fillna("None")

# Pipeline for numeric features: Iterative imputation then scale
numeric_transformer_advanced = Pipeline(steps=[
    ('impute_iterative', IterativeImputer(random_state=42)),
    ('scaler', StandardScaler())
])

# Pipeline for general categorical features: Fill missing values with 'None' then apply one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('fill_none', FunctionTransformer(fill_none, validate=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Specific transformer for 'Electrical' using the mode for imputation
electrical_transformer = Pipeline(steps=[
    ('impute_electrical', SimpleImputer(strategy='most_frequent')),
    ('onehot_electrical', OneHotEncoder(handle_unknown='ignore'))
])

# Combined preprocessor for numeric, general categorical, and electrical data
preprocessor_advanced = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_advanced, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('electrical', electrical_transformer, electrical_feature)
    ])

# Target variable
y = Ames['SalePrice']

# All features
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]

# Define the model pipelines with preprocessor and regressor
models = {
    'Lasso': Lasso(max_iter=20000),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet()
}

results_advanced = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_advanced),
        ('regressor', model)
    ])
    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y)
    results_advanced[name] = round(scores.mean(), 4)

# Output the cross-validation scores for advanced imputation
print("Cross-validation scores with Iterative Imputer:", results_advanced)

```

While the improvements in accuracy from `IterativeImputer` over `SimpleImputer` are modest, they highlight an important aspect of data imputation: the complexity and interdependencies in a dataset may not always lead to dramatically higher scores with more sophisticated methods:

```Bash
Cross-validation scores with Iterative Imputer: {'Lasso': 0.9142, 'Ridge': 0.9135, 'ElasticNet': 0.8746}
```

These modest improvements demonstrate that while `IterativeImputer` can refine the precision of our models, the extent of its impact can vary depending on the dataset’s characteristics. As we move into the third and final part of this post, we will explore `KNNImputer`, an alternative advanced technique that leverages the nearest neighbors approach, potentially offering different insights and advantages for handling missing data in various types of datasets.

## Leveraging Neighborhood Insights with KNN Imputation

In the final part of this post, we explore `KNNImputer`, which imputes missing values using the mean of the k-nearest neighbors found in the training set. This method assumes that similar data points can be found close in feature space, making it highly effective for datasets where such assumptions hold true. KNN imputation is particularly powerful in scenarios where data points with similar characteristics are likely to have similar responses or features. We examine its impact on the same predictive models, providing a full spectrum of how different imputation methods might influence the outcomes of regression analyses:

```Python
# Import the necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score

# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Exclude 'PID' and 'SalePrice' from features and specifically handle the 'Electrical' column
numeric_features = Ames.select_dtypes(include=['int64', 'float64']).drop(columns=['PID', 'SalePrice']).columns
categorical_features = Ames.select_dtypes(include=['object']).columns.difference(['Electrical'])
electrical_feature = ['Electrical']  # Specifically handle the 'Electrical' column

# Helper function to fill 'None' for missing categorical data
def fill_none(X):
    return X.fillna("None")

# Pipeline for numeric features: K-Nearest Neighbors Imputation then scale
numeric_transformer_knn = Pipeline(steps=[
    ('impute_knn', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# Pipeline for general categorical features: Fill missing values with 'None' then apply one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('fill_none', FunctionTransformer(fill_none, validate=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Specific transformer for 'Electrical' using the mode for imputation
electrical_transformer = Pipeline(steps=[
    ('impute_electrical', SimpleImputer(strategy='most_frequent')),
    ('onehot_electrical', OneHotEncoder(handle_unknown='ignore'))
])

# Combined preprocessor for numeric, general categorical, and electrical data
preprocessor_knn = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_knn, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('electrical', electrical_transformer, electrical_feature)
    ])

# Target variable
y = Ames['SalePrice']

# All features
X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]

# Define the model pipelines with preprocessor and regressor
models = {
    'Lasso': Lasso(max_iter=20000),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet()
}

results_knn = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_knn),
        ('regressor', model)
    ])
    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y)
    results_knn[name] = round(scores.mean(), 4)

# Output the cross-validation scores for KNN imputation
print("Cross-validation scores with KNN Imputer:", results_knn)
```

The cross-validation results using `KNNImputer` show a very slight improvement compared to those achieved with `SimpleImputer` and `IterativeImputer:`

```Bash
Cross-validation scores with KNN Imputer: {'Lasso': 0.9146, 'Ridge': 0.9138, 'ElasticNet': 0.8748}
```

This subtle enhancement suggests that for certain datasets, the proximity-based approach of `KNNImputer`—which factors in the similarity between data points—can be more effective in capturing and preserving the underlying structure of the data, potentially leading to more accurate predictions.

## **Further** **Reading**

#### APIs

- [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) API
- [sklearn.impute.IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) API
- [sklearn.impute.KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) API

#### Tutorials

- [Statistical Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/) by Jason Brownlee
- [Sklearn Simple Imputer Tutorial](https://youtu.be/E2i6nZEnYF0?si=-6AnWWLmWGt7wroU) by Greg Hogg

#### **Resources**

- [Ames Dataset](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)
- [Ames Data Dictionary](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **Summary**

This post has guided you through the progression from manual to automated imputation techniques, starting with a replication of basic manual imputation using `SimpleImputer` to establish a benchmark. We then explored more sophisticated strategies with `IterativeImputer`, which models each feature with missing values as dependent on other features, and concluded with `KNNImputer`, leveraging the proximity of data points to fill in missing values. Interestingly, in our case, these sophisticated techniques did not show a large improvement over the basic method. This demonstrates that while advanced imputation methods can be utilized to handle missing data, their effectiveness can vary depending on the specific characteristics and structure of the dataset involved.

Specifically, you learned:

- How to replicate and automate manual imputation processing using `SimpleImputer`.
- How improvements in predictive performance may not always justify the complexity of `IterativeImputer`.
- How `KNNImputer` demonstrates the potential for leveraging data structure in imputation, though it similarly showed only modest improvements in our dataset.

Do you have any questions? Please ask your questions in the comments below, and I will do my best to answer.

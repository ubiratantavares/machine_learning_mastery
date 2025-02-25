
# Filling the Gaps: A Comparative Guide to Imputation Techniques in Machine Learning

By [Vinod Chugani](https://machinelearningmastery.com/author/vbpm1401/ "Posts by Vinod Chugani") on September 4, 2024 in [Data Science](https://machinelearningmastery.com/category/data-science/ "View all items in Data Science") [0](https://machinelearningmastery.com/filling-the-gaps-a-comparative-guide-to-imputation-techniques-in-machine-learning/#respond)

 Share _Post_ Share

In our previous exploration of penalized regression models such as Lasso, Ridge, and ElasticNet, we demonstrated how effectively these models manage multicollinearity, allowing us to utilize a broader array of features to enhance model performance. Building on this foundation, we now address another crucial aspect of data preprocessing—handling missing values. Missing data can significantly compromise the accuracy and reliability of models if not appropriately managed. This post explores various imputation strategies to address missing data and embed them into our pipeline. This approach allows us to further refine our predictive accuracy by incorporating previously excluded features, thus making the most of our rich dataset.

Let’s get started.

![](https://machinelearningmastery.com/wp-content/uploads/2024/05/lan-deng-eAWFUVw9OX0-unsplash.jpg)

Filling the Gaps: A Comparative Guide to Imputation Techniques in Machine Learning  
Photo by [lan deng](https://unsplash.com/photos/person-in-white-shirt-and-blue-jeans-walking-inside-gap-store-eAWFUVw9OX0). Some rights reserved.

## Overview

This post is divided into three parts; they are:

- Reconstructing Manual Imputation with SimpleImputer
- Advancing Imputation Techniques with IterativeImputer
- Leveraging Neighborhood Insights with KNN Imputation

## Reconstructing Manual Imputation with SimpleImputer

In part one of this post, we revisit and reconstruct our earlier manual imputation techniques using `SimpleImputer`. Our previous exploration of the Ames Housing dataset provided foundational insights into [using the data dictionary](https://machinelearningmastery.com/classifying_variables/) to tackle missing data. We demonstrated manual imputation strategies tailored to different data types, considering domain knowledge and data dictionary details. For example, categorical variables missing in the dataset often indicate an absence of the feature (e.g., a missing ‘PoolQC’ might mean no pool exists), guiding our imputation to fill these with “None” to preserve the dataset’s integrity. Meanwhile, numerical features were handled differently, employing techniques like mean imputation.

Now, by automating these processes with scikit-learn’s `SimpleImputer`, we enhance reproducibility and efficiency. Our pipeline approach not only incorporates imputation but also scales and encodes features, preparing them for regression analysis with models such as Lasso, Ridge, and ElasticNet:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31<br><br>32<br><br>33<br><br>34<br><br>35<br><br>36<br><br>37<br><br>38<br><br>39<br><br>40<br><br>41<br><br>42<br><br>43<br><br>44<br><br>45<br><br>46<br><br>47<br><br>48<br><br>49<br><br>50<br><br>51<br><br>52<br><br>53<br><br>54<br><br>55<br><br>56<br><br>57<br><br>58<br><br>59<br><br>60<br><br>61<br><br>62<br><br>63<br><br>64<br><br>65<br><br>66<br><br>67<br><br>68<br><br>69<br><br>70<br><br>71<br><br>72|# Import the necessary libraries<br><br>import pandas as pd<br><br>from sklearn.pipeline import Pipeline<br><br>from sklearn.impute import SimpleImputer<br><br>from sklearn.compose import ColumnTransformer<br><br>from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer<br><br>from sklearn.linear_model import Lasso, Ridge, ElasticNet<br><br>from sklearn.model_selection import cross_val_score<br><br># Load the dataset<br><br>Ames = pd.read_csv('Ames.csv')<br><br># Exclude 'PID' and 'SalePrice' from features and specifically handle the 'Electrical' column<br><br>numeric_features = Ames.select_dtypes(include=['int64', 'float64']).drop(columns=['PID', 'SalePrice']).columns<br><br>categorical_features = Ames.select_dtypes(include=['object']).columns.difference(['Electrical'])<br><br>electrical_feature = ['Electrical']  # Specifically handle the 'Electrical' column<br><br># Helper function to fill 'None' for missing categorical data<br><br>def fill_none(X):<br><br>    return X.fillna("None")<br><br># Pipeline for numeric features: Impute missing values then scale<br><br>numeric_transformer = Pipeline(steps=[<br><br>    ('impute_mean', SimpleImputer(strategy='mean')),<br><br>    ('scaler', StandardScaler())<br><br>])<br><br># Pipeline for general categorical features: Fill missing values with 'None' then apply one-hot encoding<br><br>categorical_transformer = Pipeline(steps=[<br><br>    ('fill_none', FunctionTransformer(fill_none, validate=False)),<br><br>    ('onehot', OneHotEncoder(handle_unknown='ignore'))<br><br>])<br><br># Specific transformer for 'Electrical' using the mode for imputation<br><br>electrical_transformer = Pipeline(steps=[<br><br>    ('impute_electrical', SimpleImputer(strategy='most_frequent')),<br><br>    ('onehot_electrical', OneHotEncoder(handle_unknown='ignore'))<br><br>])<br><br># Combined preprocessor for numeric, general categorical, and electrical data<br><br>preprocessor = ColumnTransformer(<br><br>    transformers=[<br><br>        ('num', numeric_transformer, numeric_features),<br><br>        ('cat', categorical_transformer, categorical_features),<br><br>        ('electrical', electrical_transformer, electrical_feature)<br><br>    ])<br><br># Target variable<br><br>y = Ames['SalePrice']<br><br># All features<br><br>X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]<br><br># Define the model pipelines with preprocessor and regressor<br><br>models = {<br><br>    'Lasso': Lasso(max_iter=20000),<br><br>    'Ridge': Ridge(),<br><br>    'ElasticNet': ElasticNet()<br><br>}<br><br>results = {}<br><br>for name, model in models.items():<br><br>    pipeline = Pipeline(steps=[<br><br>        ('preprocessor', preprocessor),<br><br>        ('regressor', model)<br><br>    ])<br><br>    # Perform cross-validation<br><br>    scores = cross_val_score(pipeline, X, y)<br><br>    results[name] = round(scores.mean(), 4)<br><br># Output the cross-validation scores<br><br>print("Cross-validation scores with Simple Imputer:", results)|

The results from this implementation are displayed, showing how simple imputation affects model accuracy and establishes a benchmark for more sophisticated methods discussed later:

|   |   |
|---|---|
|1|Cross-validation scores with Simple Imputer: {'Lasso': 0.9138, 'Ridge': 0.9134, 'ElasticNet': 0.8752}|

Transitioning from manual steps to a pipeline approach using scikit-learn enhances several aspects of data processing:

1. **Efficiency and Error Reduction:** Manually imputing values is time-consuming and prone to errors, especially as data complexity increases. The pipeline automates these steps, ensuring consistent transformations and reducing mistakes.
2. **Reusability and Integration:** Manual methods are less reusable. In contrast, pipelines encapsulate the entire preprocessing and modeling steps, making them easily reusable and seamlessly integrated into the model training process.
3. **Data Leakage Prevention:** There’s a risk of data leakage with manual imputation, as it may include test data when computing values. Pipelines prevent this risk with the fit/transform methodology, ensuring calculations are derived only from the training set.

This framework, demonstrated with `SimpleImputer`, shows a flexible approach to data preprocessing that can be easily adapted to include various imputation strategies. In upcoming sections, we will explore additional techniques, assessing their impact on model performance.

## Advancing Imputation Techniques with IterativeImputer

In part two, we experiment with `IterativeImputer`, a more advanced imputation technique that models each feature with missing values as a function of other features in a round-robin fashion. Unlike simple methods that might use a general statistic such as the mean or median, Iterative Imputer models each feature with missing values as a dependent variable in a regression, informed by the other features in the dataset. This process iterates, refining estimates for missing values using the entire set of available feature interactions. This approach can unveil subtle data patterns and dependencies not captured by simpler imputation methods:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31<br><br>32<br><br>33<br><br>34<br><br>35<br><br>36<br><br>37<br><br>38<br><br>39<br><br>40<br><br>41<br><br>42<br><br>43<br><br>44<br><br>45<br><br>46<br><br>47<br><br>48<br><br>49<br><br>50<br><br>51<br><br>52<br><br>53<br><br>54<br><br>55<br><br>56<br><br>57<br><br>58<br><br>59<br><br>60<br><br>61<br><br>62<br><br>63<br><br>64<br><br>65<br><br>66<br><br>67<br><br>68<br><br>69<br><br>70<br><br>71<br><br>72<br><br>73|# Import the necessary libraries<br><br>import pandas as pd<br><br>from sklearn.pipeline import Pipeline<br><br>from sklearn.experimental import enable_iterative_imputer  # This line is needed for IterativeImputer<br><br>from sklearn.impute import SimpleImputer, IterativeImputer<br><br>from sklearn.compose import ColumnTransformer<br><br>from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer<br><br>from sklearn.linear_model import Lasso, Ridge, ElasticNet<br><br>from sklearn.model_selection import cross_val_score<br><br># Load the dataset<br><br>Ames = pd.read_csv('Ames.csv')<br><br># Exclude 'PID' and 'SalePrice' from features and specifically handle the 'Electrical' column<br><br>numeric_features = Ames.select_dtypes(include=['int64', 'float64']).drop(columns=['PID', 'SalePrice']).columns<br><br>categorical_features = Ames.select_dtypes(include=['object']).columns.difference(['Electrical'])<br><br>electrical_feature = ['Electrical']  # Specifically handle the 'Electrical' column<br><br># Helper function to fill 'None' for missing categorical data<br><br>def fill_none(X):<br><br>    return X.fillna("None")<br><br># Pipeline for numeric features: Iterative imputation then scale<br><br>numeric_transformer_advanced = Pipeline(steps=[<br><br>    ('impute_iterative', IterativeImputer(random_state=42)),<br><br>    ('scaler', StandardScaler())<br><br>])<br><br># Pipeline for general categorical features: Fill missing values with 'None' then apply one-hot encoding<br><br>categorical_transformer = Pipeline(steps=[<br><br>    ('fill_none', FunctionTransformer(fill_none, validate=False)),<br><br>    ('onehot', OneHotEncoder(handle_unknown='ignore'))<br><br>])<br><br># Specific transformer for 'Electrical' using the mode for imputation<br><br>electrical_transformer = Pipeline(steps=[<br><br>    ('impute_electrical', SimpleImputer(strategy='most_frequent')),<br><br>    ('onehot_electrical', OneHotEncoder(handle_unknown='ignore'))<br><br>])<br><br># Combined preprocessor for numeric, general categorical, and electrical data<br><br>preprocessor_advanced = ColumnTransformer(<br><br>    transformers=[<br><br>        ('num', numeric_transformer_advanced, numeric_features),<br><br>        ('cat', categorical_transformer, categorical_features),<br><br>        ('electrical', electrical_transformer, electrical_feature)<br><br>    ])<br><br># Target variable<br><br>y = Ames['SalePrice']<br><br># All features<br><br>X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]<br><br># Define the model pipelines with preprocessor and regressor<br><br>models = {<br><br>    'Lasso': Lasso(max_iter=20000),<br><br>    'Ridge': Ridge(),<br><br>    'ElasticNet': ElasticNet()<br><br>}<br><br>results_advanced = {}<br><br>for name, model in models.items():<br><br>    pipeline = Pipeline(steps=[<br><br>        ('preprocessor', preprocessor_advanced),<br><br>        ('regressor', model)<br><br>    ])<br><br>    # Perform cross-validation<br><br>    scores = cross_val_score(pipeline, X, y)<br><br>    results_advanced[name] = round(scores.mean(), 4)<br><br># Output the cross-validation scores for advanced imputation<br><br>print("Cross-validation scores with Iterative Imputer:", results_advanced)|

While the improvements in accuracy from `IterativeImputer` over `SimpleImputer` are modest, they highlight an important aspect of data imputation: the complexity and interdependencies in a dataset may not always lead to dramatically higher scores with more sophisticated methods:

|   |   |
|---|---|
|1|Cross-validation scores with Iterative Imputer: {'Lasso': 0.9142, 'Ridge': 0.9135, 'ElasticNet': 0.8746}|

These modest improvements demonstrate that while `IterativeImputer` can refine the precision of our models, the extent of its impact can vary depending on the dataset’s characteristics. As we move into the third and final part of this post, we will explore `KNNImputer`, an alternative advanced technique that leverages the nearest neighbors approach, potentially offering different insights and advantages for handling missing data in various types of datasets.

## Leveraging Neighborhood Insights with KNN Imputation

In the final part of this post, we explore `KNNImputer`, which imputes missing values using the mean of the k-nearest neighbors found in the training set. This method assumes that similar data points can be found close in feature space, making it highly effective for datasets where such assumptions hold true. KNN imputation is particularly powerful in scenarios where data points with similar characteristics are likely to have similar responses or features. We examine its impact on the same predictive models, providing a full spectrum of how different imputation methods might influence the outcomes of regression analyses:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31<br><br>32<br><br>33<br><br>34<br><br>35<br><br>36<br><br>37<br><br>38<br><br>39<br><br>40<br><br>41<br><br>42<br><br>43<br><br>44<br><br>45<br><br>46<br><br>47<br><br>48<br><br>49<br><br>50<br><br>51<br><br>52<br><br>53<br><br>54<br><br>55<br><br>56<br><br>57<br><br>58<br><br>59<br><br>60<br><br>61<br><br>62<br><br>63<br><br>64<br><br>65<br><br>66<br><br>67<br><br>68<br><br>69<br><br>70<br><br>71<br><br>72|# Import the necessary libraries<br><br>import pandas as pd<br><br>from sklearn.pipeline import Pipeline<br><br>from sklearn.impute import SimpleImputer, KNNImputer<br><br>from sklearn.compose import ColumnTransformer<br><br>from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer<br><br>from sklearn.linear_model import Lasso, Ridge, ElasticNet<br><br>from sklearn.model_selection import cross_val_score<br><br># Load the dataset<br><br>Ames = pd.read_csv('Ames.csv')<br><br># Exclude 'PID' and 'SalePrice' from features and specifically handle the 'Electrical' column<br><br>numeric_features = Ames.select_dtypes(include=['int64', 'float64']).drop(columns=['PID', 'SalePrice']).columns<br><br>categorical_features = Ames.select_dtypes(include=['object']).columns.difference(['Electrical'])<br><br>electrical_feature = ['Electrical']  # Specifically handle the 'Electrical' column<br><br># Helper function to fill 'None' for missing categorical data<br><br>def fill_none(X):<br><br>    return X.fillna("None")<br><br># Pipeline for numeric features: K-Nearest Neighbors Imputation then scale<br><br>numeric_transformer_knn = Pipeline(steps=[<br><br>    ('impute_knn', KNNImputer(n_neighbors=5)),<br><br>    ('scaler', StandardScaler())<br><br>])<br><br># Pipeline for general categorical features: Fill missing values with 'None' then apply one-hot encoding<br><br>categorical_transformer = Pipeline(steps=[<br><br>    ('fill_none', FunctionTransformer(fill_none, validate=False)),<br><br>    ('onehot', OneHotEncoder(handle_unknown='ignore'))<br><br>])<br><br># Specific transformer for 'Electrical' using the mode for imputation<br><br>electrical_transformer = Pipeline(steps=[<br><br>    ('impute_electrical', SimpleImputer(strategy='most_frequent')),<br><br>    ('onehot_electrical', OneHotEncoder(handle_unknown='ignore'))<br><br>])<br><br># Combined preprocessor for numeric, general categorical, and electrical data<br><br>preprocessor_knn = ColumnTransformer(<br><br>    transformers=[<br><br>        ('num', numeric_transformer_knn, numeric_features),<br><br>        ('cat', categorical_transformer, categorical_features),<br><br>        ('electrical', electrical_transformer, electrical_feature)<br><br>    ])<br><br># Target variable<br><br>y = Ames['SalePrice']<br><br># All features<br><br>X = Ames[numeric_features.tolist() + categorical_features.tolist() + electrical_feature]<br><br># Define the model pipelines with preprocessor and regressor<br><br>models = {<br><br>    'Lasso': Lasso(max_iter=20000),<br><br>    'Ridge': Ridge(),<br><br>    'ElasticNet': ElasticNet()<br><br>}<br><br>results_knn = {}<br><br>for name, model in models.items():<br><br>    pipeline = Pipeline(steps=[<br><br>        ('preprocessor', preprocessor_knn),<br><br>        ('regressor', model)<br><br>    ])<br><br>    # Perform cross-validation<br><br>    scores = cross_val_score(pipeline, X, y)<br><br>    results_knn[name] = round(scores.mean(), 4)<br><br># Output the cross-validation scores for KNN imputation<br><br>print("Cross-validation scores with KNN Imputer:", results_knn)|

The cross-validation results using `KNNImputer` show a very slight improvement compared to those achieved with `SimpleImputer` and `IterativeImputer:`

|   |   |
|---|---|
|1|Cross-validation scores with KNN Imputer: {'Lasso': 0.9146, 'Ridge': 0.9138, 'ElasticNet': 0.8748}|

This subtle enhancement suggests that for certain datasets, the proximity-based approach of `KNNImputer`—which factors in the similarity between data points—can be more effective in capturing and preserving the underlying structure of the data, potentially leading to more accurate predictions.

## **Further** **Reading**

#### APIs

- [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) API
- [sklearn.impute.IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) API
- [sklearn.impute.KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) API

#### Tutorials

- [Statistical Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/) by Jason Brownlee
- [Sklearn Simple Imputer Tutorial](https://youtu.be/E2i6nZEnYF0?si=-6AnWWLmWGt7wroU) by Greg Hogg

#### **Resources**

- [Ames Dataset](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)
- [Ames Data Dictionary](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **Summary**

This post has guided you through the progression from manual to automated imputation techniques, starting with a replication of basic manual imputation using `SimpleImputer` to establish a benchmark. We then explored more sophisticated strategies with `IterativeImputer`, which models each feature with missing values as dependent on other features, and concluded with `KNNImputer`, leveraging the proximity of data points to fill in missing values. Interestingly, in our case, these sophisticated techniques did not show a large improvement over the basic method. This demonstrates that while advanced imputation methods can be utilized to handle missing data, their effectiveness can vary depending on the specific characteristics and structure of the dataset involved.

Specifically, you learned:

- How to replicate and automate manual imputation processing using `SimpleImputer`.
- How improvements in predictive performance may not always justify the complexity of `IterativeImputer`.
- How `KNNImputer` demonstrates the potential for leveraging data structure in imputation, though it similarly showed only modest improvements in our dataset.

Do you have any questions? Please ask your questions in the comments below, and I will do my best to answer.

[Link](https://machinelearningmastery.com/one-hot-encoding-understanding-the-hot-in-data/)

Preparing categorical data correctly is a fundamental step in machine learning, particularly when using linear models. One Hot Encoding stands out as a key technique, enabling the transformation of categorical variables into a machine-understandable format. This post tells you why you cannot use a categorical variable directly and demonstrates the use One Hot Encoding in our search for identifying the most predictive categorical features for linear regression.

## Overview

This post is divided into three parts; they are:

- What is One Hot Encoding?
- Identifying the Most Predictive Categorical Feature
- Evaluating Individual Features’ Predictive Power

## What is One Hot Encoding?

In data preprocessing for linear models, “One Hot Encoding” is a crucial technique for managing categorical data. In this method, “hot” signifies a category’s presence (encoded as one), while “cold” (or zero) signals its absence, using binary vectors for representation.

From the angle of levels of measurement, categorical data are **nominal data**, which means if we used numbers as labels (e.g., 1 for male and 2 for female), operations such as addition and subtraction would not make sense. And if the labels are not numbers, you can’t even do any math with it.

One hot encoding separates each category of a variable into distinct features, preventing the misinterpretation of categorical data as having some ordinal significance in linear regression and other linear models. After the encoding, the number bears meaning, and it can readily be used in a math equation.

For instance, consider a categorical feature like “Color” with the values Red, Blue, and Green. One Hot Encoding translates this into three binary features (“Color_Red,” “Color_Blue,” and “Color_Green”), each indicating the presence (1) or absence (0) of a color for each observation. Such a representation clarifies to the model that these categories are distinct, with no inherent order.

Why does this matter? Many machine learning models, including linear regression, operate on numerical data and assume a numerical relationship between values. Directly encoding categories as numbers (e.g., Red=1, Blue=2, Green=3) could imply a non-existent hierarchy or quantitative relationship, potentially skewing predictions. One Hot Encoding sidesteps this issue, preserving the categorical nature of the data in a form that models can accurately interpret.

Let’s apply this technique to the Ames dataset, demonstrating the transformation process with an example:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17|# Load only categorical columns without missing values from the Ames dataset<br><br>import pandas aspd<br><br>Ames=pd.read_csv("Ames.csv").select_dtypes(include=["object"]).dropna(axis=1)<br><br>print(f"The shape of the DataFrame before One Hot Encoding is: {Ames.shape}")<br><br># Import OneHotEncoder and apply it to Ames:<br><br>from sklearn.preprocessing import OneHotEncoder<br><br>encoder=OneHotEncoder(sparse=False)<br><br>Ames_One_Hot=encoder.fit_transform(Ames)<br><br># Convert the encoded result back to a DataFrame<br><br>Ames_encoded_df=pd.DataFrame(Ames_One_Hot,columns=encoder.get_feature_names_out(Ames.columns))<br><br># Display the new DataFrame and it's expanded shape<br><br>print(Ames_encoded_df.head())<br><br>print(f"The shape of the DataFrame after One Hot Encoding is: {Ames_encoded_df.shape}")|

This will output:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11|The shape of the DataFrame before One Hot Encoding is: (2579, 27)<br><br>   MSZoning_A (agr)  ...  SaleCondition_Partial<br><br>0               0.0  ...                    0.0<br><br>1               0.0  ...                    0.0<br><br>2               0.0  ...                    0.0<br><br>3               0.0  ...                    0.0<br><br>4               0.0  ...                    0.0<br><br>[5 rows x 188 columns]<br><br>The shape of the DataFrame after One Hot Encoding is: (2579, 188)|

As seen, the Ames dataset’s categorical columns are converted into 188 distinct features, illustrating the expanded complexity and detailed representation that One Hot Encoding provides. This expansion, while increasing the dimensionality of the dataset, is a crucial preprocessing step when modeling the relationship between categorical features and the target variable in linear regression.

## Identifying the Most Predictive Categorical Feature

After understanding the basic premise and application of One Hot Encoding in linear models, the next step in our analysis involves identifying which categorical feature contributes most significantly to predicting our target variable. In the code snippet below, we iterate through each categorical feature in our dataset, apply One Hot Encoding, and evaluate its predictive power using a linear regression model in conjunction with cross-validation. Here, the `drop="first"` parameter in the `OneHotEncoder` function plays a vital role:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27|# Buidling on the code above to identify top categorical feature<br><br>from sklearn.linear_model import LinearRegression<br><br>from sklearn.model_selection import cross_val_score<br><br># Set 'SalePrice' as the target variable<br><br>y=pd.read_csv("Ames.csv")["SalePrice"]<br><br># Dictionary to store feature names and their corresponding mean CV R² scores<br><br>feature_scores={}<br><br>forfeature inAmes.columns:<br><br>    encoder=OneHotEncoder(drop="first")<br><br>    X_encoded=encoder.fit_transform(Ames[[feature]])<br><br>    # Initialize the linear regression model<br><br>    model=LinearRegression()<br><br>    # Perform 5-fold cross-validation and calculate R^2 scores<br><br>    scores=cross_val_score(model,X_encoded,y)<br><br>    mean_score=scores.mean()<br><br>    # Store the mean R^2 score<br><br>    feature_scores[feature]=mean_score<br><br># Sort features based on their mean CV R² scores in descending order<br><br>sorted_features=sorted(feature_scores.items(),key=lambda item:item[1],reverse=True)<br><br>print("Feature selected for highest predictability:",sorted_features[0][0])|

The `drop="first"` parameter is used to mitigate perfect collinearity. By dropping the first category (encoding it implicitly as zeros across all other categories for a feature), we reduce redundancy and the number of input variables without losing any information. This practice simplifies the model, making it easier to interpret and often improving its performance. The code above will output:

|   |   |
|---|---|
|1|Feature selected forhighest predictability:Neighborhood|

Our analysis reveals that “Neighborhood” is the categorical feature with the highest predictability in our dataset. This finding highlights the significant impact of location on housing prices within the Ames dataset.

## Evaluating Individual Features’ Predictive Power

With a deeper understanding of One Hot Encoding and identifying the most predictive categorical feature, we now expand our analysis to uncover the top five categorical features that significantly impact housing prices. This step is essential for fine-tuning our predictive model, enabling us to focus on the features that offer the most value in forecasting outcomes. By evaluating each feature’s mean cross-validated R² score, we can determine not just the importance of these features individually but also gain insights into how different aspects of a property contribute to its overall valuation.

Let’s delve into this evaluation:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4|# Building on the code above to determine the performance of top 5 categorical features<br><br>print("Top 5 Categorical Features:")<br><br>forfeature,score insorted_features[0:5]:<br><br>    print(f"{feature}: Mean CV R² = {score:.4f}")|

The output from our analysis presents a revealing snapshot of the factors that play pivotal roles in determining housing prices:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6|Top 5 Categorical Features:<br><br>Neighborhood: Mean CV R² = 0.5407<br><br>ExterQual: Mean CV R² = 0.4651<br><br>KitchenQual: Mean CV R² = 0.4373<br><br>Foundation: Mean CV R² = 0.2547<br><br>HeatingQC: Mean CV R² = 0.1892|

This result accentuates the importance of the feature “Neighborhood” as the top predictor, reinforcing the idea that location significantly influences housing prices. Following closely are “ExterQual” (Exterior Material Quality) and “KitchenQual” (Kitchen Quality), which highlight the premium buyers place on the quality of construction and finishes. “Foundation” and “HeatingQC” (Heating Quality and Condition) also emerge as significant, albeit with lower predictive power, suggesting that structural integrity and comfort features are critical considerations for home buyers.

## **Further****Reading**

#### APIs

- [sklearn.preprocessing.OneHotEncoder](https://12ft.io/proxy?q=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fgenerated%2Fsklearn.preprocessing.OneHotEncoder.html) API

#### Tutorials

- [One-hot encoding categorical variables](https://12ft.io/proxy?q=https%3A%2F%2Fwww.blog.trainindata.com%2Fone-hot-encoding-categorical-variables%2F) by Sole Galli

#### **Ames Housing Dataset & Data Dictionary**

- [Ames Dataset](https://12ft.io/proxy?q=https%3A%2F%2Fraw.githubusercontent.com%2FPadre-Media%2Fdataset%2Fmain%2FAmes.csv)
- [Ames Data Dictionary](https://12ft.io/proxy?q=https%3A%2F%2Fgithub.com%2FPadre-Media%2Fdataset%2Fblob%2Fmain%2FAmes%2520Data%2520Dictionary.txt)

## **Summary**

In this post, we focused on the critical process of preparing categorical data for linear models. Starting with an explanation of One Hot Encoding, we showed how this technique makes categorical data interpretable for linear regression by creating binary vectors. Our analysis identified “Neighborhood” as the categorical feature with the highest impact on housing prices, underscoring location’s pivotal role in real estate valuation.

Specifically, you learned:

- One Hot Encoding’s role in converting categorical data to a format usable by linear models, preventing the algorithm from misinterpreting the data’s nature.
- The importance of the `drop='first'` parameter in One Hot Encoding to avoid perfect collinearity in linear models.
- How to evaluate the predictive power of individual categorical features and rank their performance within the context of linear models.

Do you have any questions? Please ask your questions in the comments below, and I will do my best to answer.
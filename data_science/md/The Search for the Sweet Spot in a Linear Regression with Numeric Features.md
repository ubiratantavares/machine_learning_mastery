
[Link](https://machinelearningmastery.com/the-search-for-the-sweet-spot-in-a-linear-regression-with-numeric-features/)

Consistent with the principle of Occam’s razor, starting simple often leads to the most profound insights, especially when piecing together a predictive model. In this post, using the Ames Housing Dataset, we will first pinpoint the key features that shine on their own. Then, step by step, we’ll layer these insights, observing how their combined effect enhances our ability to forecast accurately. As we delve deeper, we will harness the power of the Sequential Feature Selector (SFS) to sift through the complexities and highlight the optimal combination of features. This methodical approach will guide us to the “sweet spot” — a harmonious blend where the selected features maximize our model’s predictive precision without overburdening it with unnecessary data.

Let’s get started.

![alt](https://machinelearningmastery.com/wp-content/uploads/2024/05/joanna-kosinska-ayOfwsd9mY-unsplash-scaled.jpg)

The Search for the Sweet Spot in a Linear Regression with Numeric Features  
Photo by [Joanna Kosinska](https://12ft.io/proxy?q=https%3A%2F%2Funsplash.com%2Fphotos%2Fassorted-color-candies-on-container--ayOfwsd9mY). Some rights reserved.

## Overview

This post is divided into three parts; they are:

- From Single Features to Collective Impact
- Diving Deeper with SFS: The Power of Combination
- Finding the Predictive “Sweet Spot”

## From Individual Strengths to Collective Impact

Our first step is to identify which features out of the myriad available in the Ames dataset stand out as powerful predictors on their own. We turn to simple linear regression models, each dedicated to one of the top standalone features identified based on their predictive power for housing prices.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29|# Load the essential libraries and Ames dataset<br><br>from sklearn.model_selection import cross_val_score<br><br>from sklearn.linear_model import LinearRegression<br><br>import pandas aspd<br><br>Ames=pd.read_csv("Ames.csv").select_dtypes(include=["int64","float64"])<br><br>Ames.dropna(axis=1,inplace=True)<br><br>X=Ames.drop("SalePrice",axis=1)<br><br>y=Ames["SalePrice"]<br><br># Initialize the Linear Regression model<br><br>model=LinearRegression()<br><br># Prepare to collect feature scores<br><br>feature_scores={}<br><br># Evaluate each feature with cross-validation<br><br>forfeature inX.columns:<br><br>    X_single=X[[feature]]<br><br>    cv_scores=cross_val_score(model,X_single,y)<br><br>    feature_scores[feature]=cv_scores.mean()<br><br># Identify the top 5 features based on mean CV R² scores<br><br>sorted_features=sorted(feature_scores.items(),key=lambda item:item[1],reverse=True)<br><br>top_5=sorted_features[0:5]<br><br># Display the top 5 features and their individual performance<br><br>forfeature,score intop_5:<br><br>    print(f"Feature: {feature}, Mean CV R²: {score:.4f}")|

This will output the top 5 features that can be used individually in a simple linear regression:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5|Feature: OverallQual, Mean CV R²: 0.6183<br><br>Feature: GrLivArea, Mean CV R²: 0.5127<br><br>Feature: 1stFlrSF, Mean CV R²: 0.3957<br><br>Feature: YearBuilt, Mean CV R²: 0.2852<br><br>Feature: FullBath, Mean CV R²: 0.2790|

Curiosity leads us further: what if we combine these top features into a single multiple linear regression model? Will their collective power surpass their individual contributions?

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11|# Extracting the top 5 features for our multiple linear regression<br><br>top_features=[feature forfeature,score intop_5]<br><br># Building the model with the top 5 features<br><br>X_top=Ames[top_features]<br><br># Evaluating the model with cross-validation<br><br>cv_scores_mlr=cross_val_score(model,X_top,y,cv=5,scoring="r2")<br><br>mean_mlr_score=cv_scores_mlr.mean()<br><br>print(f"Mean CV R² Score for Multiple Linear Regression Model: {mean_mlr_score:.4f}")|

The initial findings are promising; each feature indeed has its strengths. However, when combined in a multiple regression model, we observe a “decent” improvement—a testament to the complexity of housing price predictions.

|   |   |
|---|---|
|1|Mean CV R² Score for Multiple Linear Regression Model: 0.8003|

This result hints at untapped potential: Could there be a more strategic way to select and combine features for even greater predictive accuracy?

## Diving Deeper with SFS: The Power of Combination

As we expand our use of Sequential Feature Selector (SFS) from $n=1$ to $n=5$, an important concept comes into play: the power of combination. Let’s illustrate as we build on the code above:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11|# Perform Sequential Feature Selector with n=5 and build on above code<br><br>from sklearn.feature_selection import SequentialFeatureSelector<br><br>sfs=SequentialFeatureSelector(model,n_features_to_select=5)<br><br>sfs.fit(X,y)<br><br>selected_features=X.columns[sfs.get_support()].to_list()<br><br>print(f"Features selected by SFS: {selected_features}")<br><br>scores=cross_val_score(model,Ames[selected_features],y)<br><br>print(f"Mean CV R² Score using SFS with n=5: {scores.mean():.4f}")|

Choosing $n=5$ doesn’t merely mean selecting the five best standalone features. Rather, it’s about identifying the set of five features that, when used together, optimize the model’s predictive ability:

|   |   |
|---|---|
|1<br><br>2|Features selected by SFS: ['GrLivArea', 'OverallQual', 'YearBuilt', '1stFlrSF', 'KitchenAbvGr']<br><br>Mean CV R² Score using SFS with n=5: 0.8056|

This outcome is particularly enlightening when we compare it to the top five features selected based on their standalone predictive power. The attribute “FullBath” (not selected by SFS) was replaced by “KitchenAbvGr” in the SFS selection. This divergence highlights a fundamental principle of feature selection: **it’s the combination that counts**. SFS doesn’t just look for strong individual predictors; it seeks out features that work best in concert. This might mean selecting a feature that, on its own, wouldn’t top the list but, when combined with others, improves the model’s accuracy.

If you wonder why this is the case, the features selected in the combination should be complementary to each other rather than correlated. In this way, each new feature provides new information for the predictor instead of agreeing with what is already known.

## Finding the Predictive “Sweet Spot”

The journey to optimal feature selection begins by pushing our model to its limits. By initially considering the maximum possible features, we gain a comprehensive view of how model performance evolves by adding each feature. This visualization serves as our starting point, highlighting the diminishing returns on model predictability and guiding us toward finding the “sweet spot.” Let’s start by running a Sequential Feature Selector (SFS) across the entire feature set, plotting the performance to visualize the impact of each addition:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22|# Performance of SFS from 1 feature to maximum, building on code above:<br><br>import matplotlib.pyplot asplt<br><br># Prepare to store the mean CV R² scores for each number of features<br><br>mean_scores=[]<br><br># Iterate over a range from 1 feature to the maximum number of features available<br><br>forn_features_to_select inrange(1,len(X.columns)):<br><br>    sfs=SequentialFeatureSelector(model,n_features_to_select=n_features_to_select)<br><br>    sfs.fit(X,y)<br><br>    selected_features=X.columns[sfs.get_support()]<br><br>    score=cross_val_score(model,X[selected_features],y,cv=5,scoring="r2").mean()<br><br>    mean_scores.append(score)<br><br># Plot the mean CV R² scores against the number of features selected<br><br>plt.figure(figsize=(10,6))<br><br>plt.plot(range(1,len(X.columns)),mean_scores,marker="o")<br><br>plt.title("Performance vs. Number of Features Selected")<br><br>plt.xlabel("Number of Features")<br><br>plt.ylabel("Mean CV R² Score")<br><br>plt.grid(True)<br><br>plt.show()|

The plot below demonstrates how model performance improves as more features are added but eventually plateaus, indicating a point of diminishing returns:

[![alt](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20800%20614'%3E%3C/svg%3E)](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2F%3Fattachment_id%3D16806)

Comparing the effect of adding features to the predictor

From this plot, you can see that using more than ten features has little benefit. Using three or fewer features, however, is suboptimal. You can use the “elbow method” to find where this curve bends and determine the optimal number of features. This is a subjective decision. This plot suggests anywhere from 5 to 9 looks right.

Armed with the insights from our initial exploration, we apply a tolerance (`tol=0.005`) to our feature selection process. This can help us determine the optimal number of features objectively and robustly:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26|# Apply Sequential Feature Selector with tolerance = 0.005, building on code above<br><br>sfs_tol=SequentialFeatureSelector(model,n_features_to_select="auto",tol=0.005)<br><br>sfs_tol.fit(X,y)<br><br># Get the number of features selected with tolerance<br><br>n_features_selected=sum(sfs_tol.get_support())<br><br># Prepare to store the mean CV R² scores for each number of features<br><br>mean_scores_tol=[]<br><br># Iterate over a range from 1 feature to the Sweet Spot<br><br>forn_features_to_select inrange(1,n_features_selected+1):<br><br>    sfs=SequentialFeatureSelector(model,n_features_to_select=n_features_to_select)<br><br>    sfs.fit(X,y)<br><br>    selected_features=X.columns[sfs.get_support()]<br><br>    score=cross_val_score(model,X[selected_features],y,cv=5,scoring="r2").mean()<br><br>    mean_scores_tol.append(score)<br><br># Plot the mean CV R² scores against the number of features selected<br><br>plt.figure(figsize=(10,6))<br><br>plt.plot(range(1,n_features_selected+1),mean_scores_tol,marker="o")<br><br>plt.title("The Sweet Spot: Performance vs. Number of Features Selected")<br><br>plt.xlabel("Number of Features")<br><br>plt.ylabel("Mean CV R² Score")<br><br>plt.grid(True)<br><br>plt.show()|

This strategic move allows us to concentrate on those features that provide the highest predictability, culminating in the selection of 8 optimal features:

[![alt](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20800%20614'%3E%3C/svg%3E)](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2F%3Fattachment_id%3D16808)

Finding the optimal number of features from a plot

We can now conclude our findings by showing the features selected by SFS:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5|# Print the selected features and their performance, building on the above:<br><br>selected_features=X.columns[sfs_tol.get_support()]<br><br>print(f"Number of features selected: {n_features_selected}")<br><br>print(f"Selected features: {selected_features.tolist()}")<br><br>print(f"Mean CV R² Score using SFS with tol=0.005: {mean_scores_tol[-1]:.4f}")|

|   |   |
|---|---|
|1<br><br>2<br><br>3|Number of features selected: 8<br><br>Selected features: ['GrLivArea', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', '1stFlrSF', 'BedroomAbvGr', 'KitchenAbvGr']<br><br>Mean CV R² Score using SFS with tol=0.005: 0.8239|

By focusing on these 8 features, we achieve a model that balances complexity with high predictability, showcasing the effectiveness of a measured approach to feature selection.

## **Further****Reading**

#### APIs

- [sklearn.feature_selection.SequentialFeatureSelector](https://12ft.io/proxy?q=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fgenerated%2Fsklearn.feature_selection.SequentialFeatureSelector.html) API

#### Tutorials

- [Sequential Feature Selection](https://12ft.io/proxy?q=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D0vCXcGJg5Bo) by Sebastian Raschka

#### **Ames Housing Dataset & Data Dictionary**

- [Ames Dataset](https://12ft.io/proxy?q=https%3A%2F%2Fraw.githubusercontent.com%2FPadre-Media%2Fdataset%2Fmain%2FAmes.csv)
- [Ames Data Dictionary](https://12ft.io/proxy?q=https%3A%2F%2Fgithub.com%2FPadre-Media%2Fdataset%2Fblob%2Fmain%2FAmes%2520Data%2520Dictionary.txt)

## **Summary**

Through this three-part post, you have embarked on a journey from assessing the predictive power of individual features to harnessing their combined strength in a refined model. Our exploration has demonstrated that while more features can enhance a model’s ability to capture complex patterns, there comes a point where additional features no longer contribute to improved predictions. By applying a tolerance level to the Sequential Feature Selector, you have honed in on an optimal set of features that propel our model’s performance to its peak without overcomplicating the predictive landscape. This sweet spot—identified as eight key features—epitomizes the strategic melding of simplicity and sophistication in predictive modeling.

Specifically, you learned:

- **The Art of Starting Simple**: Beginning with simple linear regression models to understand each feature’s standalone predictive value sets the foundation for more complex analyses.
- **Synergy in Selection**: The transition to the Sequential Feature Selector underscores the importance of not just individual feature strengths but their synergistic impact when combined effectively.
- **Maximizing Model Efficacy**: The quest for the predictive sweet spot through SFS with a set tolerance teaches us the value of precision in feature selection, achieving the most with the least.

Do you have any questions? Please ask your questions in the comments below, and I will do my best to answer.


[[Data Science|Data Science]]

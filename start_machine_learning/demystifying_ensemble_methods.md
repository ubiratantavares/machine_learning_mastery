# Demystifying Ensemble Methods: Boosting, Bagging, and Stacking Explained
By Iván Palomares Carrascosa on November 23, 2024 in Start Machine Learning 0
 Post Share
Demystifying Ensemble Methods: Boosting, Bagging, and Stacking Explained
Demystifying Ensemble Methods: Boosting, Bagging, and Stacking Explained
Image by Editor | Ideogram

Unity makes strength. This well-known motto perfectly captures the essence of ensemble methods: one of the most powerful machine learning (ML) approaches -with permission from deep neural networks- to effectively address complex problems predicated on complex data, by combining multiple models for addressing one predictive task. This article describes three common ways to build ensemble models: boosting, bagging, and stacking. Let’s get started!

Bagging
Bagging involves training multiple models independently and in parallel. The models are usually of the same type, for instance, a set of decision trees or polynomial regressors. The difference between each model is that each one is trained on a random subset of the whole training data. After each model returns a prediction, all predictions are aggregated into one overall prediction. How? It depends on the type of predictive task:

For a bagging ensemble of regression models, numerical predictions are averaged.
For a bagging ensemble of classification models, class predictions are combined by majority vote.
In both cases, aggregating multiple model predictions reduces variance and improves overall performance, compared to standalone ML models.

Random data selection in bagging can be instance-based or attribute-based:

In instance-based bagging, models are trained on random subsets of data instances, typically sampled with replacement through a process called bootstrapping. Sampling by replacement means that one particular instance in the dataset could be randomly chosen for none, one, or more than one of training the models that will become part of the ensemble.
In attribute-based bagging, each model in the ensemble uses a different random subset of features in the training data, thereby introducing diversity among the models. This approach helps alleviate the so-called curse of dimensionality: a problem found when training ML models on datasets with a very large number of features, resulting in loss of efficiency, possible overfitting (the model learns excessively from the data and it memorizes it, losing the ability to generalize to future data), and so on.
The randomness in the two selection processes described above helps the ensemble method learn more comprehensively about different “regions” of the data while avoiding overfitting, ultimately making the system more robust.

Illustration of a bagging ensemble
Illustration of a bagging ensemble
Image by Author

Random forests are a widely used example of a bagging method that combines both instance and attribute-level randomness. As its name suggests, a random forest builds multiple decision trees, each trained on a bootstrapped sample of the data and a random subset of features per tree. This twofold sampling promotes diversity among the trees and reduces the correlation between models.


Boosting
Unlike bagging ensembles where multiple models are trained in parallel and their individual predictions are aggregated, boosting adopts a sequential approach. In boosting ensembles, several models of the same type are trained one after another, each one correcting the most noticeable errors made by the previous model. As errors get gradually fixed by several models one after another, the ensemble eventually produces a stronger overall solution that is more accurate and robust against complex patterns in the data.

Illustration of a boosting ensemble
Illustration of a boosting ensemble
Image by Author

XGBoost (Extreme Gradient Boosting) is a popular example of a boosting-based ensemble. XGBoost builds models sequentially, focusing heavily on correcting errors at each step, and is known for its efficiency, speed, and high performance in competitive machine learning tasks. Although not strictly limited to decision trees, XGBoost resembles random forests because it is designed to operate particularly well on ensembles of decision trees.


Stacking
A slightly more complex approach is stacking, which often combines different types of models (like decision tree classifiers, logistic regression classifiers, and neural networks put together), trained separately on the same data. The catch: each type of model typically captures patterns in the data differently. Moreover, instead of aggregating individual predictions, stacking goes one step further: individual predictions are used as inputs to a final-stage ML model, called meta-model, which learns to weigh and combine predictions of the base models as if they were data instances. In sum, the combined strengths of each specific model’s inference skills lead to a more accurate final decision.

Illustration of a stacking ensemble
Illustration of a stacking ensemble
Image by Author

Stacked Generalization is a common stacking approach, where the meta-model is often a simple linear or logistic regression model.


Wrapping Up
Ensemble methods like boosting, bagging, and stacking leverage the strengths of combining multiple ML models to enhance predictive accuracy and robustness. The unique properties of each approach will help you tackle complex data challenges more successfully, turning possible individual model weaknesses into collective strengths.



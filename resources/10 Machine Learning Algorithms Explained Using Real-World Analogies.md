
By [Kanwal Mehreen](https://machinelearningmastery.com/author/kanwalmehreen/ "Posts by Kanwal Mehreen") on September 3, 2024 in [Resources](https://machinelearningmastery.com/category/resources/ "View all items in Resources") [11](https://machinelearningmastery.com/10-machine-learning-algorithms-explained-using-real-world-analogies/)

When I was in high school and studied complex mathematics problems, I always used to think about why we were studying them or why they were useful. I was unable to understand and find their usage in the real world. Since machine learning is also a trending topic that many people want to explore, the complex mathematics and abstraction behind machine learning algorithms make it difficult for beginners to appreciate and learn its usage.

This is where **analogous learning** comes into play. It allows you to associate real-world analogies with complex concepts that help you to stay curious and think creatively. This really helps when you actually apply these algorithms to solve real-world problems later on. Taking motivation from this approach, I will explain 10 common machine learning algorithms by associating them with real-world analogies, so let’s get started.

## 1. Linear Regression

Linear regression is a supervised machine learning algorithm that tries to fit the best straight line between your features and your target variable so that the difference between the actual and the predicted value is as minimal as possible.

**Analogy:** Suppose you are a gardener and you want to test what is an ideal amount of fertilizer that you should be giving to a plant to increase its growth. So for that purpose, you record the amount of fertilizer (feature/independent variable) and also measure the corresponding plant’s growth (dependent or target variable) for one month. Now, you have the information and you plot it on a scatter plot and try to figure out the best straight line that passes through these points in such a way that the deviation of all the points from the line is minimum. Once you have this line, you can predict future plant growth based on the amount of fertilizer used.

## 2. Logistic Regression

It is somewhat similar to linear regression and a supervised learning problem, but linear regression predicts the continuous target variable while logistic regression is used for binary classification problems where it predicts the probability of a binary outcome like the probability of yes or no, true or false.

**Analogy:** For example, if you watch America’s Got Talent, we know that there are only 2 choices: either the candidate will be selected for the next round or not. So, you consider various factors like their current performance, past experience, whether it’s a unique act or not, and based on this, you decide how much capability the candidate has to be successful in the next round. Based on this, you either press the red (reject) button on or the green (accept) one.

## 3. Decision Tree

Decision tree is a supervised learning algorithm that recursively divides your data into subsets based on the feature values. Each division on the node is actually a decision that decides the direction of the traversal and helps in making predictions.

**Analogy:** Have you ever played that “20 Questions” game with your friend? That’s exactly how decision trees work. Let me share what happens in this game. So, your friend thinks of something that you have to guess and all you can do is ask them yes/no questions to narrow down the possible answers. Each answer helps you make a decision and eliminates options until you guess the correct answer.

## 4. Random Forest Algorithm

Random forest is an ensemble learning technique that uses multiple decision trees trained on various parts of the data. It then combines the predictions by each tree to make a final decision.

**Analogy:** Consider a committee of investors and equity holders at a business who need to make a decision regarding a new deal. Everyone has their own thought process and experiences. Based on their analysis, each of them provides their judgment. In the end, all the judgments are combined to make the final decision.

## 5. Support Vector Machine (SVM)

SVM is a supervised machine learning algorithm that divides the classes using a straight line (hyperplane) in such a way that the distance between them is maximum. When a new data point comes in, it’s easier to identify which group/class it belongs to.

**Analogy:** If you’re a sports lover, you would understand the rivalry between the fans of 2 opposing teams on the ground. So, you try to separate these 2 groups as far as possible by maybe, let’s say, tying a red ribbon, and when a new person joins in, based on features like the shirt they are wearing or which team they support, you may provide the seating arrangement accordingly.

## 6. Naive Bayes Algorithm

It is a supervised machine learning algorithm based on Bayes’ Theorem and assumes that the features are independent. It computes the probability of each class keeping in mind some prior information, and then the class with the highest probability is selected. It is mainly used for classification problems.

**Analogy:** We are all aware of spam emails, right? So basically, the filter looks out for some common words like “free”, “discount,” “limited time,” or “click here,” without considering the context in which these words appear. Although this may classify some cases incorrectly, it saves a huge amount of time when it comes to processing tons of emails. It considers these spam words as independent features to determine the likelihood of an email being spam or not.

## 7. K-Nearest Neighbors (KNN) Algorithm

KNN is a supervised learning algorithm that assumes similar data points will be closer to each other in feature space (just like close friends sitting together in class). It makes prediction about the label of an unknown data point using its K nearest known neighbors, where it is a hyperparameter and represents the number of voting neighbours.

**Analogy:** Let’s say you want to try some new restaurants and have a couple of options. So you ask your friends (neighbors in KNN) for recommendations. Each of them recommends their favorite place that they have visited, and the place that gets the majority vote is then what you finalize to visit.

## 8. K-means

K-means is an unsupervised learning algorithm that assigns data points to unique clusters based on their position. It starts by randomly initializing the centroids and calculating the distance of each point to these centroids. Each point is then assigned to the cluster of the nearest centroid. The new data points within each cluster are averaged to find new centroids. This process repeats until the centroids no longer change, meaning that the data points have been perfectly classified into clusters.

**Analogy:** Consider that you are part of a reading community and they have to create 3 groups of, let’s say, 18 students. Initially, they will assign them randomly into 3 groups of 6 people. Then in the 2nd iteration, they reassign based on their interests gathered from a form. Then after their interaction, they make changes unless the final groups are created with people that have aligned interests.

## 9. Principal Component Analysis

PCA is an unsupervised learning algorithm. It is a dimensionality reduction technique that identifies the principal (important) components of the data and maps it to a lower-dimensional space making it easier to analyze.

**Analogy:** Most of us have traveled somewhere and I know it’s a headache to pack things. Let’s say we just have one suitcase, so what do we do? We start by first filling it in with important items and then try to find a way to compress the less important ones or remove them from our bag. That’s exactly how PCA works, identifying important features and condensing or removing the less relevant ones.

## 10. Gradient Boosting

It is an ensemble learning algorithm that combines multiple weak models to create a strong model. It works iteratively where each model tries to improve the errors made by the previous model, increasing the overall performance.

**Analogy:** Has it ever happened that you were initially struggling with a subject but gradually improved? What’s the general approach typically followed in that situation? You take a test, receive the grades, and then work on your shortcomings or the topics where you lack understanding. This process gradually improves your overall performance in that subject.

This brings me to the end of my article. I really enjoy explaining these concepts using analogies, and I hope you find them as helpful as I do. Let me know in the comments if you enjoyed this approach or if you have suggestions for other topics!

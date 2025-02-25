# How to Improve Performance With Transfer Learning for Deep Learning Neural Networks

By[Jason Brownlee](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fauthor%2Fjasonb%2F "Posts by Jason Brownlee")onAugust 25, 2020in[Deep Learning Performance](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fcategory%2Fbetter-deep-learning%2F "View all items in Deep Learning Performance") [51](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fhow-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks%2F%23comments)

Share_Post_Share

An interesting benefit of deep learning neural networks is that they can be reused on related problems.

[Transfer learning](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftransfer-learning-for-deep-learning%2F) refers to a technique for predictive modeling on a different but somehow similar problem that can then be reused partly or wholly to accelerate the training and improve the performance of a model on the problem of interest.

In deep learning, this means reusing the weights in one or more layers from a pre-trained network model in a new model and either keeping the weights fixed, fine tuning them, or adapting the weights entirely when training the model.

In this tutorial, you will discover how to use transfer learning to improve the performance deep learning neural networks in Python with Keras.

After completing this tutorial, you will know:

- Transfer learning is a method for reusing a model trained on a related predictive modeling problem.
- Transfer learning can be used to accelerate the training of neural networks as either a weight initialization scheme or feature extraction method.
- How to use transfer learning to improve the performance of an MLP for a multiclass classification problem.

**Kick-start your project** with my new book [Better Deep Learning](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fbetter-deep-learning%2F), including _step-by-step tutorials_ and the _Python source code_ files for all examples.

Let’s get started.

- **Updated Oct/2019**: Updated for Keras 2.3 and TensorFlow 2.0.
- **Update Jan/2020**: Updated for changes in scikit-learn v0.22 API.

![How to Improve Performance With Transfer Learning for Deep Learning Neural Networks](https://machinelearningmastery.com/wp-content/uploads/2019/02/How-to-Improve-Performance-With-Transfer-Learning-for-Deep-Learning-Neural-Networks.jpg)

How to Improve Performance With Transfer Learning for Deep Learning Neural Networks  
Photo by [Damian Gadal](https://12ft.io/proxy?q=https%3A%2F%2Fwww.flickr.com%2Fphotos%2F23024164%40N06%2F13885404633%2F), some rights reserved.

## Tutorial Overview

This tutorial is divided into six parts; they are:

1. What Is Transfer Learning?
2. Blobs Multi-Class Classification Problem
3. Multilayer Perceptron Model for Problem 1
4. Standalone MLP Model for Problem 2
5. MLP With Transfer Learning for Problem 2
6. Comparison of Models on Problem 2

## What Is Transfer Learning?

Transfer learning generally refers to a process where a model trained on one problem is used in some way on a second related problem.

> Transfer learning and domain adaptation refer to the situation where what has been learned in one setting (i.e., distribution P1) is exploited to improve generalization in another setting (say distribution P2).

— Page 536, [Deep Learning](https://12ft.io/proxy?q=https%3A%2F%2Famzn.to%2F2NJW3gE), 2016.

In deep learning, transfer learning is a technique whereby a neural network model is first trained on a problem similar to the problem that is being solved. One or more layers from the trained model are then used in a new model trained on the problem of interest.

> This is typically understood in a supervised learning context, where the input is the same but the target may be of a different nature. For example, we may learn about one set of visual categories, such as cats and dogs, in the first setting, then learn about a different set of visual categories, such as ants and wasps, in the second setting.

— Page 536, [Deep Learning](https://12ft.io/proxy?q=https%3A%2F%2Famzn.to%2F2NJW3gE), 2016.

Transfer learning has the benefit of decreasing the training time for a neural network model and resulting in lower generalization error.

There are two main approaches to implementing transfer learning; they are:

- Weight Initialization.
- Feature Extraction.

The weights in re-used layers may be used as the starting point for the training process and adapted in response to the new problem. This usage treats transfer learning as a type of weight initialization scheme. This may be useful when the first related problem has a lot more labeled data than the problem of interest and the similarity in the structure of the problem may be useful in both contexts.

> … the objective is to take advantage of data from the first setting to extract information that may be useful when learning or even when directly making predictions in the second setting.

— Page 538, [Deep Learning](https://12ft.io/proxy?q=https%3A%2F%2Famzn.to%2F2NJW3gE), 2016.

Alternately, the weights of the network may not be adapted in response to the new problem, and only new layers after the reused layers may be trained to interpret their output. This usage treats transfer learning as a type of feature extraction scheme. An example of this approach is the re-use of deep convolutional neural network models trained for photo classification as feature extractors when developing [photo captioning models](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdevelop-a-deep-learning-caption-generation-model-in-python%2F).

Variations on these usages may involve not training the weights of the model on the new problem initially, but later fine tuning all weights of the learned model with a [small learning rate](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Flearning-rate-for-deep-learning-neural-networks%2F).

### Want Better Results with Deep Learning?

Take my free 7-day email crash course now (with sample code).

Click to sign-up and also get a free PDF Ebook version of the course.

Download Your FREE Mini-Course

## Blobs Multi-Class Classification Problem

We will use a small multi-class classification problem as the basis to demonstrate transfer learning.

The scikit-learn class provides the [make_blobs() function](https://12ft.io/proxy?q=http%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fgenerated%2Fsklearn.datasets.make_blobs.html) that can be used to create a multi-class classification problem with the prescribed number of samples, input variables, classes, and variance of samples within a class.

We can configure the problem to have two input variables (to represent the _x_ and _y_ coordinates of the points) and a standard deviation of 2.0 for points within each group. We will use the same random state (seed for the pseudorandom number generator) to ensure that we always get the same data points.

|   |   |
|---|---|
|1<br><br>2|# generate 2d classification dataset<br><br>X,y=make_blobs(n_samples=1000,centers=3,n_features=2,cluster_std=2,random_state=1)|

The results are the input and output elements of a dataset that we can model.

The “_random_state_” argument can be varied to give different versions of the problem (different cluster centers). We can use this to generate samples from two different problems: train a model on one problem and re-use the weights to better learn a model for a second problem.

Specifically, we will refer to _random_state=1_ as Problem 1 and _random_state=2_ as Problem 2.

- **Problem 1**. Blobs problem with two input variables and three classes with the _random_state_ argument set to one.
- **Problem 2**. Blobs problem with two input variables and three classes with the _random_state_ argument set to two.

In order to get a feeling for the complexity of the problem, we can plot each point on a two-dimensional scatter plot and color each point by class value.

The complete example is listed below.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31|# plot of blobs multiclass classification problems 1 and 2<br><br>from sklearn.datasets import make_blobs<br><br>from numpy import where<br><br>from matplotlib import pyplot<br><br># generate samples for blobs problem with a given random seed<br><br>def samples_for_seed(seed):<br><br># generate samples<br><br>X,y=make_blobs(n_samples=1000,centers=3,n_features=2,cluster_std=2,random_state=seed)<br><br>returnX,y<br><br># create a scatter plot of points colored by class value<br><br>def plot_samples(X,y,classes=3):<br><br># plot points for each class<br><br>foriinrange(classes):<br><br># select indices of points with each class label<br><br>samples_ix=where(y==i)<br><br># plot points for this class with a given color<br><br>pyplot.scatter(X[samples_ix,0],X[samples_ix,1])<br><br># generate multiple problems<br><br>n_problems=2<br><br>foriinrange(1,n_problems+1):<br><br># specify subplot<br><br>pyplot.subplot(210+i)<br><br># generate samples<br><br>X,y=samples_for_seed(i)<br><br># scatter plot of samples<br><br>plot_samples(X,y)<br><br># plot figure<br><br>pyplot.show()|

Running the example generates a sample of 1,000 examples for Problem 1 and Problem 2 and creates a scatter plot for each sample, coloring the data points by their class value.

![Scatter Plots of Blobs Dataset for Problems 1 and 2 With Three Classes and Points Colored by Class Value](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%201280%20960'%3E%3C/svg%3E)

Scatter Plots of Blobs Dataset for Problems 1 and 2 With Three Classes and Points Colored by Class Value

This provides a good basis for transfer learning as each version of the problem has similar input data with a similar scale, although with different target information (e.g. cluster centers).

We would expect that aspects of a model fit on one version of the blobs problem (e.g. Problem 1) to be useful when fitting a model on a new version of the blobs problem (e.g. Problem 2).

## Multilayer Perceptron Model for Problem 1

In this section, we will develop a Multilayer Perceptron model (MLP) for Problem 1 and [save the model to file](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fsave-load-keras-deep-learning-models%2F) so that we can reuse the weights later.

First, we will develop a function to prepare the dataset ready for modeling. After the make_blobs() function is called with a given random seed (e.g, one in this case for Problem 1), the target variable must be one hot encoded so that we can develop a model that predicts the probability of a given sample belonging to each of the target classes.

The prepared samples can then be split in half, with 500 examples for both the train and test datasets. The _samples_for_seed()_ function below implements this, preparing the dataset for a given random number seed and re-tuning the train and test sets split into input and output components.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11|# prepare a blobs examples with a given random seed<br><br>def samples_for_seed(seed):<br><br># generate samples<br><br>X,y=make_blobs(n_samples=1000,centers=3,n_features=2,cluster_std=2,random_state=seed)<br><br># one hot encode output variable<br><br>y=to_categorical(y)<br><br># split into train and test<br><br>n_train=500<br><br>trainX,testX=X[:n_train,:],X[n_train:,:]<br><br>trainy,testy=y[:n_train],y[n_train:]<br><br>returntrainX,trainy,testX,testy|

We can call this function to prepare a dataset for Problem 1 as follows.

|   |   |
|---|---|
|1<br><br>2|# prepare data<br><br>trainX,trainy,testX,testy=samples_for_seed(1)|

Next, we can define and fit a model on the training dataset.

The model will expect two inputs for the two variables in the data. The model will have two hidden layers with five nodes each and the rectified linear activation function. Two layers are probably not required for this function, although we’re interested in the model learning some deep structure that we can reuse across instances of this problem. The output layer has three nodes, one for each class in the target variable and the softmax activation function.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5|# define model<br><br>model=Sequential()<br><br>model.add(Dense(5,input_dim=2,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(5,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(3,activation='softmax'))|

Given that the problem is a multi-class classification problem, the categorical cross-entropy loss function is minimized and the stochastic gradient descent with the default learning rate and no momentum is used to learn the problem.

|   |   |
|---|---|
|1<br><br>2|# compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])|

The model is fit for 100 epochs on the training dataset and the test set is used as a validation dataset during training, evaluating the performance on both datasets at the end of each epoch so that we can [plot learning curves](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fhow-to-control-neural-network-model-capacity-with-nodes-and-layers%2F).

|   |   |
|---|---|
|1|history=model.fit(trainX,trainy,validation_data=(testX,testy),epochs=100,verbose=0)|

The _fit_model()_ function ties these elements together, taking the train and test datasets as arguments and returning the fit model and training history.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12|# define and fit model on a training dataset<br><br>def fit_model(trainX,trainy,testX,testy):<br><br># define model<br><br>model=Sequential()<br><br>model.add(Dense(5,input_dim=2,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(5,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(3,activation='softmax'))<br><br># compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])<br><br># fit model<br><br>history=model.fit(trainX,trainy,validation_data=(testX,testy),epochs=100,verbose=0)<br><br>returnmodel,history|

We can call this function with the prepared dataset to obtain a fit model and the history collected during the training process.

|   |   |
|---|---|
|1<br><br>2|# fit model on train dataset<br><br>model,history=fit_model(trainX,trainy,testX,testy)|

Finally, we can summarize the performance of the model.

The classification accuracy of the model on the train and test sets can be evaluated.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4|# evaluate the model<br><br>_,train_acc=model.evaluate(trainX,trainy,verbose=0)<br><br>_,test_acc=model.evaluate(testX,testy,verbose=0)<br><br>print('Train: %.3f, Test: %.3f'%(train_acc,test_acc))|

The history collected during training can be used to create line plots showing both the loss and classification accuracy for the model on the train and test sets over each training epoch, providing learning curves.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13|# plot loss during training<br><br>pyplot.subplot(211)<br><br>pyplot.title('Loss')<br><br>pyplot.plot(history.history['loss'],label='train')<br><br>pyplot.plot(history.history['val_loss'],label='test')<br><br>pyplot.legend()<br><br># plot accuracy during training<br><br>pyplot.subplot(212)<br><br>pyplot.title('Accuracy')<br><br>pyplot.plot(history.history['accuracy'],label='train')<br><br>pyplot.plot(history.history['val_accuracy'],label='test')<br><br>pyplot.legend()<br><br>pyplot.show()|

The _summarize_model()_ function below implements this, taking the fit model, training history, and dataset as arguments and printing the model performance and creating a plot of model learning curves.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19|# summarize the performance of the fit model<br><br>def summarize_model(model,history,trainX,trainy,testX,testy):<br><br># evaluate the model<br><br>_,train_acc=model.evaluate(trainX,trainy,verbose=0)<br><br>_,test_acc=model.evaluate(testX,testy,verbose=0)<br><br>print('Train: %.3f, Test: %.3f'%(train_acc,test_acc))<br><br># plot loss during training<br><br>pyplot.subplot(211)<br><br>pyplot.title('Loss')<br><br>pyplot.plot(history.history['loss'],label='train')<br><br>pyplot.plot(history.history['val_loss'],label='test')<br><br>pyplot.legend()<br><br># plot accuracy during training<br><br>pyplot.subplot(212)<br><br>pyplot.title('Accuracy')<br><br>pyplot.plot(history.history['accuracy'],label='train')<br><br>pyplot.plot(history.history['val_accuracy'],label='test')<br><br>pyplot.legend()<br><br>pyplot.show()|

We can call this function with the fit model and prepared data.

|   |   |
|---|---|
|1<br><br>2|# evaluate model behavior<br><br>summarize_model(model,history,trainX,trainy,testX,testy)|

At the end of the run, we can save the model to file so that we may load it later and use it as the basis for some transfer learning experiments.

Note that saving the model to file requires that you have the _h5py_ library installed. This library can be installed via _pip_ as follows:

|   |   |
|---|---|
|1|sudo pip install h5py|

The fit model can be saved by calling the _save()_ function on the model.

|   |   |
|---|---|
|1<br><br>2|# save model to file<br><br>model.save('model.h5')|

Tying these elements together, the complete example of fitting an MLP on Problem 1, summarizing the model’s performance, and saving the model to file is listed below.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31<br><br>32<br><br>33<br><br>34<br><br>35<br><br>36<br><br>37<br><br>38<br><br>39<br><br>40<br><br>41<br><br>42<br><br>43<br><br>44<br><br>45<br><br>46<br><br>47<br><br>48<br><br>49<br><br>50<br><br>51<br><br>52<br><br>53<br><br>54<br><br>55<br><br>56<br><br>57<br><br>58<br><br>59<br><br>60<br><br>61|# fit mlp model on problem 1 and save model to file<br><br>from sklearn.datasets import make_blobs<br><br>from keras.layers import Dense<br><br>from keras.models import Sequential<br><br>from keras.optimizers import SGD<br><br>from keras.utils import to_categorical<br><br>from matplotlib import pyplot<br><br># prepare a blobs examples with a given random seed<br><br>def samples_for_seed(seed):<br><br># generate samples<br><br>X,y=make_blobs(n_samples=1000,centers=3,n_features=2,cluster_std=2,random_state=seed)<br><br># one hot encode output variable<br><br>y=to_categorical(y)<br><br># split into train and test<br><br>n_train=500<br><br>trainX,testX=X[:n_train,:],X[n_train:,:]<br><br>trainy,testy=y[:n_train],y[n_train:]<br><br>returntrainX,trainy,testX,testy<br><br># define and fit model on a training dataset<br><br>def fit_model(trainX,trainy,testX,testy):<br><br># define model<br><br>model=Sequential()<br><br>model.add(Dense(5,input_dim=2,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(5,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(3,activation='softmax'))<br><br># compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])<br><br># fit model<br><br>history=model.fit(trainX,trainy,validation_data=(testX,testy),epochs=100,verbose=0)<br><br>returnmodel,history<br><br># summarize the performance of the fit model<br><br>def summarize_model(model,history,trainX,trainy,testX,testy):<br><br># evaluate the model<br><br>_,train_acc=model.evaluate(trainX,trainy,verbose=0)<br><br>_,test_acc=model.evaluate(testX,testy,verbose=0)<br><br>print('Train: %.3f, Test: %.3f'%(train_acc,test_acc))<br><br># plot loss during training<br><br>pyplot.subplot(211)<br><br>pyplot.title('Loss')<br><br>pyplot.plot(history.history['loss'],label='train')<br><br>pyplot.plot(history.history['val_loss'],label='test')<br><br>pyplot.legend()<br><br># plot accuracy during training<br><br>pyplot.subplot(212)<br><br>pyplot.title('Accuracy')<br><br>pyplot.plot(history.history['accuracy'],label='train')<br><br>pyplot.plot(history.history['val_accuracy'],label='test')<br><br>pyplot.legend()<br><br>pyplot.show()<br><br># prepare data<br><br>trainX,trainy,testX,testy=samples_for_seed(1)<br><br># fit model on train dataset<br><br>model,history=fit_model(trainX,trainy,testX,testy)<br><br># evaluate model behavior<br><br>summarize_model(model,history,trainX,trainy,testX,testy)<br><br># save model to file<br><br>model.save('model.h5')|

Running the example fits and evaluates the performance of the model, printing the classification accuracy on the train and test sets.

**Note**: Your [results may vary](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdifferent-results-each-time-in-machine-learning%2F) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

In this case, we can see that the model performed well on Problem 1, achieving a classification accuracy of about 92% on both the train and test datasets.

|   |   |
|---|---|
|1|Train: 0.916, Test: 0.920|

A figure is also created summarizing the learning curves of the model, showing both the loss (top) and accuracy (bottom) for the model on both the train (blue) and test (orange) datasets at the end of each training epoch.

Your plot may not look identical but is expected to show the same general behavior. If not, try running the example a few times.

In this case, we can see that the model learned the problem reasonably quickly and well, perhaps converging in about 40 epochs and remaining reasonably stable on both datasets.

![Loss and Accuracy Learning Curves on the Train and Test Sets for an MLP on Problem 1](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%201280%20960'%3E%3C/svg%3E)

Loss and Accuracy Learning Curves on the Train and Test Sets for an MLP on Problem 1

Now that we have seen how to develop a standalone MLP for the blobs Problem 1, we can look at the doing the same for Problem 2 that can be used as a baseline.

## Standalone MLP Model for Problem 2

The example in the previous section can be updated to fit an MLP model to Problem 2.

It is important to get an idea of performance and learning dynamics on Problem 2 for a standalone model first as this will provide a baseline in performance that can be used to compare to a model fit on the same problem using transfer learning.

A single change is required that changes the call to _samples_for_seed()_ to use the pseudorandom number generator seed of two instead of one.

|   |   |
|---|---|
|1<br><br>2|# prepare data<br><br>trainX,trainy,testX,testy=samples_for_seed(2)|

For completeness, the full example with this change is listed below.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31<br><br>32<br><br>33<br><br>34<br><br>35<br><br>36<br><br>37<br><br>38<br><br>39<br><br>40<br><br>41<br><br>42<br><br>43<br><br>44<br><br>45<br><br>46<br><br>47<br><br>48<br><br>49<br><br>50<br><br>51<br><br>52<br><br>53<br><br>54<br><br>55<br><br>56<br><br>57<br><br>58<br><br>59|# fit mlp model on problem 2 and save model to file<br><br>from sklearn.datasets import make_blobs<br><br>from keras.layers import Dense<br><br>from keras.models import Sequential<br><br>from keras.optimizers import SGD<br><br>from keras.utils import to_categorical<br><br>from matplotlib import pyplot<br><br># prepare a blobs examples with a given random seed<br><br>def samples_for_seed(seed):<br><br># generate samples<br><br>X,y=make_blobs(n_samples=1000,centers=3,n_features=2,cluster_std=2,random_state=seed)<br><br># one hot encode output variable<br><br>y=to_categorical(y)<br><br># split into train and test<br><br>n_train=500<br><br>trainX,testX=X[:n_train,:],X[n_train:,:]<br><br>trainy,testy=y[:n_train],y[n_train:]<br><br>returntrainX,trainy,testX,testy<br><br># define and fit model on a training dataset<br><br>def fit_model(trainX,trainy,testX,testy):<br><br># define model<br><br>model=Sequential()<br><br>model.add(Dense(5,input_dim=2,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(5,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(3,activation='softmax'))<br><br># compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])<br><br># fit model<br><br>history=model.fit(trainX,trainy,validation_data=(testX,testy),epochs=100,verbose=0)<br><br>returnmodel,history<br><br># summarize the performance of the fit model<br><br>def summarize_model(model,history,trainX,trainy,testX,testy):<br><br># evaluate the model<br><br>_,train_acc=model.evaluate(trainX,trainy,verbose=0)<br><br>_,test_acc=model.evaluate(testX,testy,verbose=0)<br><br>print('Train: %.3f, Test: %.3f'%(train_acc,test_acc))<br><br># plot loss during training<br><br>pyplot.subplot(211)<br><br>pyplot.title('Loss')<br><br>pyplot.plot(history.history['loss'],label='train')<br><br>pyplot.plot(history.history['val_loss'],label='test')<br><br>pyplot.legend()<br><br># plot accuracy during training<br><br>pyplot.subplot(212)<br><br>pyplot.title('Accuracy')<br><br>pyplot.plot(history.history['accuracy'],label='train')<br><br>pyplot.plot(history.history['val_accuracy'],label='test')<br><br>pyplot.legend()<br><br>pyplot.show()<br><br># prepare data<br><br>trainX,trainy,testX,testy=samples_for_seed(2)<br><br># fit model on train dataset<br><br>model,history=fit_model(trainX,trainy,testX,testy)<br><br># evaluate model behavior<br><br>summarize_model(model,history,trainX,trainy,testX,testy)|

Running the example fits and evaluates the performance of the model, printing the classification accuracy on the train and test sets.

**Note**: Your [results may vary](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdifferent-results-each-time-in-machine-learning%2F) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

In this case, we can see that the model performed okay on Problem 2, but not as well as was seen on Problem 1, achieving a classification accuracy of about 79% on both the train and test datasets.

|   |   |
|---|---|
|1|Train: 0.794, Test: 0.794|

A figure is also created summarizing the learning curves of the model. Your plot may not look identical but is expected to show the same general behavior. If not, try running the example a few times.

In this case, we can see that the model converged more slowly than we saw on Problem 1 in the previous section. This suggests that this version of the problem may be slightly more challenging, at least for the chosen model configuration.

![Loss and Accuracy Learning Curves on the Train and Test Sets for an MLP on Problem 2](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%201280%20960'%3E%3C/svg%3E)

Loss and Accuracy Learning Curves on the Train and Test Sets for an MLP on Problem 2

Now that we have a baseline of performance and learning dynamics for an MLP on Problem 2, we can see how the addition of transfer learning affects the MLP on this problem.

## MLP With Transfer Learning for Problem 2

The model that was fit on Problem 1 can be loaded and the weights can be used as the initial weights for a model fit on Problem 2.

This is a type of transfer learning where learning on a different but related problem is used as a type of weight initialization scheme.

This requires that the _fit_model()_ function be updated to load the model and refit it on examples for Problem 2.

The model saved in ‘model.h5’ can be loaded using the _load_model()_ Keras function.

|   |   |
|---|---|
|1<br><br>2|# load model<br><br>model=load_model('model.h5')|

Once loaded, the model can be compiled and fit as per normal.

The updated _fit_model()_ with this change is listed below.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9|# load and re-fit model on a training dataset<br><br>def fit_model(trainX,trainy,testX,testy):<br><br># load model<br><br>model=load_model('model.h5')<br><br># compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])<br><br># re-fit model<br><br>history=model.fit(trainX,trainy,validation_data=(testX,testy),epochs=100,verbose=0)<br><br>returnmodel,history|

We would expect that a model that uses the weights from a model fit on a different but related problem to learn the problem perhaps faster in terms of the learning curve and perhaps result in lower generalization error, although these aspects would be dependent on the choice of problems and model.

For completeness, the full example with this change is listed below.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31<br><br>32<br><br>33<br><br>34<br><br>35<br><br>36<br><br>37<br><br>38<br><br>39<br><br>40<br><br>41<br><br>42<br><br>43<br><br>44<br><br>45<br><br>46<br><br>47<br><br>48<br><br>49<br><br>50<br><br>51<br><br>52<br><br>53<br><br>54<br><br>55<br><br>56<br><br>57|# transfer learning with mlp model on problem 2<br><br>from sklearn.datasets import make_blobs<br><br>from keras.layers import Dense<br><br>from keras.models import Sequential<br><br>from keras.optimizers import SGD<br><br>from keras.utils import to_categorical<br><br>from keras.models import load_model<br><br>from matplotlib import pyplot<br><br># prepare a blobs examples with a given random seed<br><br>def samples_for_seed(seed):<br><br># generate samples<br><br>X,y=make_blobs(n_samples=1000,centers=3,n_features=2,cluster_std=2,random_state=seed)<br><br># one hot encode output variable<br><br>y=to_categorical(y)<br><br># split into train and test<br><br>n_train=500<br><br>trainX,testX=X[:n_train,:],X[n_train:,:]<br><br>trainy,testy=y[:n_train],y[n_train:]<br><br>returntrainX,trainy,testX,testy<br><br># load and re-fit model on a training dataset<br><br>def fit_model(trainX,trainy,testX,testy):<br><br># load model<br><br>model=load_model('model.h5')<br><br># compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])<br><br># re-fit model<br><br>history=model.fit(trainX,trainy,validation_data=(testX,testy),epochs=100,verbose=0)<br><br>returnmodel,history<br><br># summarize the performance of the fit model<br><br>def summarize_model(model,history,trainX,trainy,testX,testy):<br><br># evaluate the model<br><br>_,train_acc=model.evaluate(trainX,trainy,verbose=0)<br><br>_,test_acc=model.evaluate(testX,testy,verbose=0)<br><br>print('Train: %.3f, Test: %.3f'%(train_acc,test_acc))<br><br># plot loss during training<br><br>pyplot.subplot(211)<br><br>pyplot.title('Loss')<br><br>pyplot.plot(history.history['loss'],label='train')<br><br>pyplot.plot(history.history['val_loss'],label='test')<br><br>pyplot.legend()<br><br># plot accuracy during training<br><br>pyplot.subplot(212)<br><br>pyplot.title('Accuracy')<br><br>pyplot.plot(history.history['accuracy'],label='train')<br><br>pyplot.plot(history.history['val_accuracy'],label='test')<br><br>pyplot.legend()<br><br>pyplot.show()<br><br># prepare data<br><br>trainX,trainy,testX,testy=samples_for_seed(2)<br><br># fit model on train dataset<br><br>model,history=fit_model(trainX,trainy,testX,testy)<br><br># evaluate model behavior<br><br>summarize_model(model,history,trainX,trainy,testX,testy)|

Running the example fits and evaluates the performance of the model, printing the classification accuracy on the train and test sets.

**Note**: Your [results may vary](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdifferent-results-each-time-in-machine-learning%2F) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

In this case, we can see that the model achieved a lower generalization error, achieving an accuracy of about 81% on the test dataset for Problem 2 as compared to the standalone model that achieved about 79% accuracy.

|   |   |
|---|---|
|1|Train: 0.786, Test: 0.810|

A figure is also created summarizing the learning curves of the model. Your plot may not look identical but is expected to show the same general behavior. If not, try running the example a few times.

In this case, we can see that the model does appear to have a similar learning curve, although we do see apparent improvements in the learning curve for the test set (orange line) both in terms of better performance earlier (epoch 20 onward) and above the performance of the model on the training set.

![Loss and Accuracy Learning Curves on the Train and Test Sets for an MLP With Transfer Learning on Problem 2](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%201280%20960'%3E%3C/svg%3E)

Loss and Accuracy Learning Curves on the Train and Test Sets for an MLP With Transfer Learning on Problem 2

We have only looked at single runs of a standalone MLP model and an MLP with transfer learning.

Neural network algorithms are stochastic, therefore an average of performance across multiple runs is required to see if the observed behavior is real or a statistical fluke.

## Comparison of Models on Problem 2

In order to determine whether using transfer learning for the blobs multi-class classification problem has a real effect, we must repeat each experiment multiple times and analyze the average performance across the repeats.

We will compare the performance of the standalone model trained on Problem 2 to a model using transfer learning, averaged over 30 repeats.

Further, we will investigate whether keeping the weights in some of the layers fixed improves model performance.

The model trained on Problem 1 has two hidden layers. By keeping the first or the first and second hidden layers fixed, the layers with unchangeable weights will act as a feature extractor and may provide features that make learning Problem 2 easier, affecting the speed of learning and/or the accuracy of the model on the test set.

As the first step, we will simplify the _fit_model()_ function to fit the model and discard any training history so that we can focus on the final accuracy of the trained model.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12|# define and fit model on a training dataset<br><br>def fit_model(trainX,trainy):<br><br># define model<br><br>model=Sequential()<br><br>model.add(Dense(5,input_dim=2,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(5,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(3,activation='softmax'))<br><br># compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])<br><br># fit model<br><br>model.fit(trainX,trainy,epochs=100,verbose=0)<br><br>returnmodel|

Next, we can develop a function that will repeatedly fit a new standalone model on Problem 2 on the training dataset and evaluate accuracy on the test set.

The _eval_standalone_model()_ function below implements this, taking the train and test sets as arguments as well as the number of repeats and returns a list of accuracy scores for models on the test dataset.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10|# repeated evaluation of a standalone model<br><br>def eval_standalone_model(trainX,trainy,testX,testy,n_repeats):<br><br>scores=list()<br><br>for_inrange(n_repeats):<br><br># define and fit a new model on the train dataset<br><br>model=fit_model(trainX,trainy)<br><br># evaluate model on test dataset<br><br>_,test_acc=model.evaluate(testX,testy,verbose=0)<br><br>scores.append(test_acc)<br><br>returnscores|

Summarizing the distribution of accuracy scores returned from this function will give an idea of how well the chosen standalone model performs on Problem 2.

|   |   |
|---|---|
|1<br><br>2<br><br>3|# repeated evaluation of standalone model<br><br>standalone_scores=eval_standalone_model(trainX,trainy,testX,testy,n_repeats)<br><br>print('Standalone %.3f (%.3f)'%(mean(standalone_scores),std(standalone_scores)))|

Next, we need an equivalent function for evaluating a model using transfer learning.

In each loop, the model trained on Problem 1 must be loaded from file, fit on the training dataset for Problem 2, then evaluated on the test set for Problem 2.

In addition, we will configure 0, 1, or 2 of the hidden layers in the loaded model to remain fixed. Keeping 0 hidden layers fixed means that all of the weights in the model will be adapted when learning Problem 2, using transfer learning as a weight initialization scheme. Whereas, keeping both (2) of the hidden layers fixed means that only the output layer of the model will be adapted during training, using transfer learning as a feature extraction method.

The _eval_transfer_model()_ function below implements this, taking the train and test datasets for Problem 2 as arguments as well as the number of hidden layers in the loaded model to keep fixed and the number of times to repeat the experiment.

The function returns a list of test accuracy scores and summarizing this distribution will give a reasonable idea of how well the model with the chosen type of transfer learning performs on Problem 2.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17|# repeated evaluation of a model with transfer learning<br><br>def eval_transfer_model(trainX,trainy,testX,testy,n_fixed,n_repeats):<br><br>scores=list()<br><br>for_inrange(n_repeats):<br><br># load model<br><br>model=load_model('model.h5')<br><br># mark layer weights as fixed or not trainable<br><br>foriinrange(n_fixed):<br><br>model.layers[i].trainable=False<br><br># re-compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])<br><br># fit model on train dataset<br><br>model.fit(trainX,trainy,epochs=100,verbose=0)<br><br># evaluate model on test dataset<br><br>_,test_acc=model.evaluate(testX,testy,verbose=0)<br><br>scores.append(test_acc)<br><br>returnscores|

We can call this function repeatedly, setting n_fixed to 0, 1, 2 in a loop and summarizing performance as we go; for example:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5|# repeated evaluation of transfer learning model, vary fixed layers<br><br>n_fixed=3<br><br>foriinrange(n_fixed):<br><br>scores=eval_transfer_model(trainX,trainy,testX,testy,i,n_repeats)<br><br>print('Transfer (fixed=%d) %.3f (%.3f)'%(i,mean(scores),std(scores)))|

In addition to reporting the mean and standard deviation of each model, we can collect all scores and create a box and whisker plot to summarize and compare the distributions of model scores.

Tying all of the these elements together, the complete example is listed below.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31<br><br>32<br><br>33<br><br>34<br><br>35<br><br>36<br><br>37<br><br>38<br><br>39<br><br>40<br><br>41<br><br>42<br><br>43<br><br>44<br><br>45<br><br>46<br><br>47<br><br>48<br><br>49<br><br>50<br><br>51<br><br>52<br><br>53<br><br>54<br><br>55<br><br>56<br><br>57<br><br>58<br><br>59<br><br>60<br><br>61<br><br>62<br><br>63<br><br>64<br><br>65<br><br>66<br><br>67<br><br>68<br><br>69<br><br>70<br><br>71<br><br>72<br><br>73<br><br>74<br><br>75<br><br>76<br><br>77<br><br>78<br><br>79<br><br>80<br><br>81<br><br>82<br><br>83<br><br>84<br><br>85<br><br>86<br><br>87|# compare standalone mlp model performance to transfer learning<br><br>from sklearn.datasets import make_blobs<br><br>from keras.layers import Dense<br><br>from keras.models import Sequential<br><br>from keras.optimizers import SGD<br><br>from keras.utils import to_categorical<br><br>from keras.models import load_model<br><br>from matplotlib import pyplot<br><br>from numpy import mean<br><br>from numpy import std<br><br># prepare a blobs examples with a given random seed<br><br>def samples_for_seed(seed):<br><br># generate samples<br><br>X,y=make_blobs(n_samples=1000,centers=3,n_features=2,cluster_std=2,random_state=seed)<br><br># one hot encode output variable<br><br>y=to_categorical(y)<br><br># split into train and test<br><br>n_train=500<br><br>trainX,testX=X[:n_train,:],X[n_train:,:]<br><br>trainy,testy=y[:n_train],y[n_train:]<br><br>returntrainX,trainy,testX,testy<br><br># define and fit model on a training dataset<br><br>def fit_model(trainX,trainy):<br><br># define model<br><br>model=Sequential()<br><br>model.add(Dense(5,input_dim=2,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(5,activation='relu',kernel_initializer='he_uniform'))<br><br>model.add(Dense(3,activation='softmax'))<br><br># compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])<br><br># fit model<br><br>model.fit(trainX,trainy,epochs=100,verbose=0)<br><br>returnmodel<br><br># repeated evaluation of a standalone model<br><br>def eval_standalone_model(trainX,trainy,testX,testy,n_repeats):<br><br>scores=list()<br><br>for_inrange(n_repeats):<br><br># define and fit a new model on the train dataset<br><br>model=fit_model(trainX,trainy)<br><br># evaluate model on test dataset<br><br>_,test_acc=model.evaluate(testX,testy,verbose=0)<br><br>scores.append(test_acc)<br><br>returnscores<br><br># repeated evaluation of a model with transfer learning<br><br>def eval_transfer_model(trainX,trainy,testX,testy,n_fixed,n_repeats):<br><br>scores=list()<br><br>for_inrange(n_repeats):<br><br># load model<br><br>model=load_model('model.h5')<br><br># mark layer weights as fixed or not trainable<br><br>foriinrange(n_fixed):<br><br>model.layers[i].trainable=False<br><br># re-compile model<br><br>model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])<br><br># fit model on train dataset<br><br>model.fit(trainX,trainy,epochs=100,verbose=0)<br><br># evaluate model on test dataset<br><br>_,test_acc=model.evaluate(testX,testy,verbose=0)<br><br>scores.append(test_acc)<br><br>returnscores<br><br># prepare data for problem 2<br><br>trainX,trainy,testX,testy=samples_for_seed(2)<br><br>n_repeats=30<br><br>dists,dist_labels=list(),list()<br><br># repeated evaluation of standalone model<br><br>standalone_scores=eval_standalone_model(trainX,trainy,testX,testy,n_repeats)<br><br>print('Standalone %.3f (%.3f)'%(mean(standalone_scores),std(standalone_scores)))<br><br>dists.append(standalone_scores)<br><br>dist_labels.append('standalone')<br><br># repeated evaluation of transfer learning model, vary fixed layers<br><br>n_fixed=3<br><br>foriinrange(n_fixed):<br><br>scores=eval_transfer_model(trainX,trainy,testX,testy,i,n_repeats)<br><br>print('Transfer (fixed=%d) %.3f (%.3f)'%(i,mean(scores),std(scores)))<br><br>dists.append(scores)<br><br>dist_labels.append('transfer f='+str(i))<br><br># box and whisker plot of score distributions<br><br>pyplot.boxplot(dists,labels=dist_labels)<br><br>pyplot.show()|

Running the example first reports the mean and standard deviation of classification accuracy on the test dataset for each model.

**Note**: Your [results may vary](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdifferent-results-each-time-in-machine-learning%2F) given the stochastic nature of the algorithm or evaluation procedure, or differences in numerical precision. Consider running the example a few times and compare the average outcome.

In this case, we can see that the standalone model achieved an accuracy of about 78% on Problem 2 with a large standard deviation of 10%. In contrast, we can see that the spread of all of the transfer learning models is much smaller, ranging from about 0.05% to 1.5%.

This difference in the standard deviations of the test accuracy scores shows the stability that transfer learning can bring to the model, reducing the variance in the performance of the final model introduced via the stochastic learning algorithm.

Comparing the mean test accuracy of the models, we can see that transfer learning that used the model as a weight initialization scheme (fixed=0) resulted in better performance than the standalone model with about 80% accuracy.

Keeping all hidden layers fixed (fixed=2) and using them as a feature extraction scheme resulted in worse performance on average than the standalone model. It suggests that the approach is too restrictive in this case.

Interestingly, we see best performance when the first hidden layer is kept fixed (fixed=1) and the second hidden layer is adapted to the problem with a test classification accuracy of about 81%. This suggests that in this case, the problem benefits from both the feature extraction and weight initialization properties of transfer learning.

It may be interesting to see how results of this last approach compare to the same model where the weights of the second hidden layer (and perhaps the output layer) are re-initialized with random numbers. This comparison would demonstrate whether the feature extraction properties of transfer learning alone or both feature extraction and weight initialization properties are beneficial.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4|Standalone 0.787 (0.101)<br><br>Transfer (fixed=0) 0.805 (0.004)<br><br>Transfer (fixed=1) 0.817 (0.005)<br><br>Transfer (fixed=2) 0.750 (0.014)|

A figure is created showing four box and whisker plots. The box shows the middle 50% of each data distribution, the orange line shows the median, and the dots show outliers.

The boxplot for the standalone model shows a number of outliers, indicating that on average, the model performs well, but there is a chance that it can perform very poorly.

Conversely, we see that the behavior of the models with transfer learning are more stable, showing a tighter distribution in performance.

![Box and Whisker Plot Comparing Standalone and Transfer Learning Models via Test Set Accuracy on the Blobs Multiclass Classification Problem](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%201280%20960'%3E%3C/svg%3E)

Box and Whisker Plot Comparing Standalone and Transfer Learning Models via Test Set Accuracy on the Blobs Multiclass Classification Problem

## Extensions

This section lists some ideas for extending the tutorial that you may wish to explore.

- **Reverse Experiment**. Train and save a model for Problem 2 and see if it can help when using it for transfer learning on Problem 1.
- **Add Hidden Layer**. Update the example to keep both hidden layers fixed, but add a new hidden layer with randomly initialized weights after the fixed layers before the output layer and compare performance.
- **Randomly Initialize Layers**. Update the example to randomly initialize the weights of the second hidden layer and the output layer and compare performance.

If you explore any of these extensions, I’d love to know.

## Further Reading

This section provides more resources on the topic if you are looking to go deeper.

### Posts

- [A Gentle Introduction to Transfer Learning for Deep Learning](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftransfer-learning-for-deep-learning%2F)
- [How to Develop a Deep Learning Photo Caption Generator from Scratch](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdevelop-a-deep-learning-caption-generation-model-in-python%2F)

### Papers

- [Deep Learning of Representations for Unsupervised and Transfer Learning](https://12ft.io/proxy?q=http%3A%2F%2Fproceedings.mlr.press%2Fv27%2Fbengio12a.html), 2011.
- [Domain Adaptation for Large-Scale Sentiment Classification: A Deep Learning Approach](https://12ft.io/proxy?q=https%3A%2F%2Fdl.acm.org%2Fcitation.cfm%3Fid%3D3104547), 2011.
- [Is Learning The n-th Thing Any Easier Than Learning The First?](https://12ft.io/proxy?q=http%3A%2F%2Fpapers.nips.cc%2Fpaper%2F1034-is-learning-the-n-th-thing-any-easier-than-learning-the-first.pdf), 1996.

### Books

- Section 15.2 Transfer Learning and Domain Adaptation, [Deep Learning](https://12ft.io/proxy?q=https%3A%2F%2Famzn.to%2F2NJW3gE), 2016.

### Articles

- [Transfer learning, Wikipedia](https://12ft.io/proxy?q=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FTransfer_learning)

## Summary

In this tutorial, you discovered how to use transfer learning to improve the performance deep learning neural networks in Python with Keras.

Specifically, you learned:

- Transfer learning is a method for reusing a model trained on a related predictive modeling problem.
- Transfer learning can be used to accelerate the training of neural networks as either a weight initialization scheme or feature extraction method.
- How to use transfer learning to improve the performance of an MLP for a multiclass classification problem.

Do you have any questions?  
Ask your questions in the comments below and I will do my best to answer.
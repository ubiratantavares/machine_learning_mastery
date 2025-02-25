# Gentle Introduction to Predictive Modeling

By[Jason Brownlee](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fauthor%2Fjasonb%2F "Posts by Jason Brownlee")onJuly 22, 2020in[Start Machine Learning](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fcategory%2Fstart-machine-learning%2F "View all items in Start Machine Learning") [79](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fgentle-introduction-to-predictive-modeling%2F%23comments)

Share_Post_Share

When you’re an absolute beginner it can be very confusing. Frustratingly so.

Even ideas that seem so simple in retrospect are alien when you first encounter them. There’s a whole new language to learn.

I recently received this question:

> So using the [iris exercise](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fhow-to-run-your-first-classifier-in-weka%2F) as an example if I were to pluck a flower from my garden how would I use the algorithm to predict what it is?

It’s a great question.

In this post I want to give a gentle introduction to predictive modeling.

![How to Develop an Auxiliary Classifier GAN (AC-GAN) From Scratch with Keras](https://machinelearningmastery.com/wp-content/uploads/2019/07/How-to-Develop-an-Auxiliary-Classifier-GAN-AC-GAN-From-Scratch-with-Keras.jpg)

Gentle Introduction to Predictive Modeling

## 1. Sample Data

Data is information about the problem that you are working on.

Imagine we want to identify the species of flower from the measurements of a flower.

The data is comprised of four flower measurements in centimeters, these are the columns of the data.

Each row of data is one example of a flower that has been measured and it’s known species.

The problem we are solving is to create a model from the sample data that can tell us which species a flower belongs to from its measurements alone.

![Sample of Iris flower data](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20552%20213'%3E%3C/svg%3E)

Sample of Iris flower data

## 2. Learn a Model

This problem described above is called supervised learning.

The goal of a supervised learning algorithm is to take some data with a known relationship (actual flower measurements and the species of the flower) and to create a model of those relationships.

In this case the output is a category (flower species) and we call this type of problem a classification problem. If the output was a numerical value, we would call it a regression problem.

The algorithm does the learning. The model contains the learned relationships.

The model itself may be a handful of numbers and a way of using those numbers to relate input (flower measurements in centimeters) to an output (the species of flower).

We want to keep the model after we have learned it from our sample data.

[![Create a Predictive Model](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20423%20335'%3E%3C/svg%3E)](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fwp-content%2Fuploads%2F2015%2F09%2FCreate-a-Predictive-Model.png)

Create a predictive model from training data and an algorithm.

## 3. Make Predictions

We don’t need to keep the training data as the model has summarized the relationships contained within it.

The reason we keep the model learned from data is because we want to use it to make predictions.

In this example, we use the model by taking measurements of specific flowers of which don’t know the species.

Our model will read the input (new measurements), perform a calculation of some kind with it’s internal numbers and make a prediction about which species of flower it happens to be.

The prediction may not be perfect, but if you have good sample data and a robust model learned from that data, it will be quite accurate.

![Make Predictions](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20591%20180'%3E%3C/svg%3E)

Use the model to make predictions on new data.

## Summary

In this post we have taken a very gentle introduction to predictive modeling.

The three aspects of predictive modeling we looked at were:

1. **Sample Data**: the data that we collect that describes our problem with known relationships between inputs and outputs.
2. **Learn a Model**: the algorithm that we use on the sample data to create a model that we can later use over and over again.
3. **Making Predictions**: the use of our learned model on new data for which we don’t know the output.

We used the example of classifying plant species based on flower measurements.

This is in fact a [famous example](https://12ft.io/proxy?q=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FIris_flower_data_set) in machine learning because it’s a good clean dataset and the problem is easy to understand.

## Action Step

Take a moment and really understand these concepts.

They are the foundation of any thinking or work that you might do in machine learning.

Your action step is to think through the three aspects (data, model, predictions) and relate them to a problem that you would like to work on.

Any questions at all, please ask in the comments. I’m here to help.
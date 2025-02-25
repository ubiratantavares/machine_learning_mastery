# How to Use The Pre-Trained VGG Model to Classify Objects in Photographs

By[Jason Brownlee](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fauthor%2Fjasonb%2F "Posts by Jason Brownlee")onAugust 19, 2019in[Deep Learning](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fcategory%2Fdeep-learning%2F "View all items in Deep Learning") [205](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fuse-pre-trained-vgg-model-classify-objects-photographs%2F%23comments)

Share_Post_Share

Convolutional neural networks are now capable of outperforming humans on some computer vision tasks, such as classifying images.

That is, given a photograph of an object, answer the question as to which of 1,000 specific objects the photograph shows.

A competition-winning model for this task is the VGG model by researchers at Oxford. What is important about this model, besides its capability of classifying objects in photographs, is that the model weights are freely available and can be loaded and used in your own models and applications.

In this tutorial, you will discover the VGG convolutional neural network models for image classification.

After completing this tutorial, you will know:

- About the ImageNet dataset and competition and the VGG winning models.
- How to load the VGG model in Keras and summarize its structure.
- How to use the loaded VGG model to classifying objects in ad hoc photographs.

**Kick-start your project** with my new book [Deep Learning With Python](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fdeep-learning-with-python%2F), including _step-by-step tutorials_ and the _Python source code_ files for all examples.

Let’s get started.

## Tutorial Overview

This tutorial is divided into 4 parts; they are:

1. ImageNet
2. The Oxford VGG Models
3. Load the VGG Model in Keras
4. Develop a Simple Photo Classifier

## ImageNet

[ImageNet](https://12ft.io/proxy?q=http%3A%2F%2Fwww.image-net.org%2F) is a research project to develop a large database of images with annotations, e.g. images and their descriptions.

The images and their annotations have been the basis for an image classification challenge called the [ImageNet Large Scale Visual Recognition Challenge](https://12ft.io/proxy?q=http%3A%2F%2Fwww.image-net.org%2Fchallenges%2FLSVRC%2F) or ILSVRC since 2010. The result is that research organizations battle it out on pre-defined datasets to see who has the best model for classifying the objects in images.

> The ImageNet Large Scale Visual Recognition Challenge is a benchmark in object category classification and detection on hundreds of object categories and millions of images. The challenge has been run annually from 2010 to present, attracting participation from more than fifty institutions.

— [ImageNet Large Scale Visual Recognition Challenge](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1409.0575), 2015.

For the classification task, images must be classified into one of 1,000 different categories.

For the last few years very deep convolutional neural network models have been used to win these challenges and results on the tasks have exceeded human performance.

![Sample of Images from the ImageNet Dataset used in the ILSVRC Challenge](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%201360%201294'%3E%3C/svg%3E)

Sample of Images from the ImageNet Dataset used in the ILSVRC Challenge  
Taken From “ImageNet Large Scale Visual Recognition Challenge”, 2015.

## The Oxford VGG Models

Researchers from the [Oxford Visual Geometry Group](https://12ft.io/proxy?q=http%3A%2F%2Fwww.robots.ox.ac.uk%2F~vgg%2F), or VGG for short, participate in the ILSVRC challenge.

In 2014, convolutional neural network models (CNN) developed by the VGG [won the image classification tasks](https://12ft.io/proxy?q=http%3A%2F%2Fimage-net.org%2Fchallenges%2FLSVRC%2F2014%2Fresults).

![ILSVRC Results in 2014 for the Classification task](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%201666%20700'%3E%3C/svg%3E)

ILSVRC Results in 2014 for the Classification task

After the competition, the participants wrote up their findings in the paper:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1409.1556), 2014.

They also made their models and learned weights [available online](https://12ft.io/proxy?q=http%3A%2F%2Fwww.robots.ox.ac.uk%2F~vgg%2Fresearch%2Fvery_deep%2F).

This allowed other researchers and developers to use a state-of-the-art image classification model in their own work and programs.

This helped to fuel a rash of [transfer learning](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Ftransfer-learning-for-deep-learning%2F) work where pre-trained models are used with minor modification on wholly new predictive modeling tasks, harnessing the state-of-the-art feature extraction capabilities of proven models.

> … we come up with significantly more accurate ConvNet architectures, which not only achieve the state-of-the-art accuracy on ILSVRC classification and localisation tasks, but are also applicable to other image recognition datasets, where they achieve excellent performance even when used as a part of a relatively simple pipelines (e.g. deep features classified by a linear SVM without fine-tuning). We have released our two best-performing models to facilitate further research.

— [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1409.1556), 2014.

VGG released two different CNN models, specifically a 16-layer model and a 19-layer model.

Refer to the paper for the full details of these models.

The VGG models are not longer state-of-the-art by only a few percentage points. Nevertheless, they are very powerful models and useful both as image classifiers and as the basis for new models that use image inputs.

In the next section, we will see how we can use the VGG model directly in Keras.

## Load the VGG Model in Keras

The VGG model can be loaded and used in the Keras deep learning library.

Keras provides an [Applications interface](https://12ft.io/proxy?q=https%3A%2F%2Fkeras.io%2Fapplications%2F) for loading and using pre-trained models.

Using this interface, you can create a VGG model using the pre-trained weights provided by the Oxford group and use it as a starting point in your own model, or use it as a model directly for classifying images.

In this tutorial, we will focus on the use case of classifying new images using the VGG model.

Keras provides both the 16-layer and 19-layer version via the VGG16 and VGG19 classes. Let’s focus on the VGG16 model.

The model can be created as follows:

|   |   |
|---|---|
|1<br><br>2|from keras.applications.vgg16 import VGG16<br><br>model=VGG16()|

That’s it.

The first time you run this example, Keras will download the weight files from the Internet and store them in the _~/.keras/models_ directory.

**Note** that the weights are about 528 megabytes, so the download may take a few minutes depending on the speed of your Internet connection.

The weights are only downloaded once. The next time you run the example, the weights are loaded locally and the model should be ready to use in seconds.

We can use the standard Keras tools for inspecting the model structure.

For example, you can print a summary of the network layers as follows:

|   |   |
|---|---|
|1<br><br>2<br><br>3|from keras.applications.vgg16 import VGG16<br><br>model=VGG16()<br><br>print(model.summary())|

You can see that the model is huge.

You can also see that, by default, the model expects images as input with the size 224 x 224 pixels with 3 channels (e.g. color).

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31<br><br>32<br><br>33<br><br>34<br><br>35<br><br>36<br><br>37<br><br>38<br><br>39<br><br>40<br><br>41<br><br>42<br><br>43<br><br>44<br><br>45<br><br>46<br><br>47<br><br>48<br><br>49<br><br>50<br><br>51<br><br>52<br><br>53|_________________________________________________________________<br><br>Layer (type)                 Output Shape              Param #<br><br>=================================================================<br><br>input_1 (InputLayer)         (None, 224, 224, 3)       0<br><br>_________________________________________________________________<br><br>block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792<br><br>_________________________________________________________________<br><br>block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928<br><br>_________________________________________________________________<br><br>block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0<br><br>_________________________________________________________________<br><br>block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856<br><br>_________________________________________________________________<br><br>block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584<br><br>_________________________________________________________________<br><br>block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0<br><br>_________________________________________________________________<br><br>block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168<br><br>_________________________________________________________________<br><br>block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080<br><br>_________________________________________________________________<br><br>block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080<br><br>_________________________________________________________________<br><br>block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0<br><br>_________________________________________________________________<br><br>block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160<br><br>_________________________________________________________________<br><br>block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808<br><br>_________________________________________________________________<br><br>block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808<br><br>_________________________________________________________________<br><br>block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0<br><br>_________________________________________________________________<br><br>block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808<br><br>_________________________________________________________________<br><br>block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808<br><br>_________________________________________________________________<br><br>block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808<br><br>_________________________________________________________________<br><br>block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0<br><br>_________________________________________________________________<br><br>flatten (Flatten)            (None, 25088)             0<br><br>_________________________________________________________________<br><br>fc1 (Dense)                  (None, 4096)              102764544<br><br>_________________________________________________________________<br><br>fc2 (Dense)                  (None, 4096)              16781312<br><br>_________________________________________________________________<br><br>predictions (Dense)          (None, 1000)              4097000<br><br>=================================================================<br><br>Total params: 138,357,544<br><br>Trainable params: 138,357,544<br><br>Non-trainable params: 0<br><br>_________________________________________________________________|

We can also create a plot of the layers in the VGG model, as follows:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4|from keras.applications.vgg16 import VGG16<br><br>from keras.utils.vis_utils import plot_model<br><br>model=VGG16()<br><br>plot_model(model,to_file='vgg.png')|

Again, because the model is large, the plot is a little too large and perhaps unreadable. Nevertheless, it is provided below.

![Plot of Layers in the VGG Model](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20251%202201'%3E%3C/svg%3E)

Plot of Layers in the VGG Model

The _VGG()_ class takes a few arguments that may only interest you if you are looking to use the model in your own project, e.g. for transfer learning.

For example:

- **include_top** (_True_): Whether or not to include the output layers for the model. You don’t need these if you are fitting the model on your own problem.
- **weights** (‘_imagenet_‘): What weights to load. You can specify None to not load pre-trained weights if you are interested in training the model yourself from scratch.
- **input_tensor** (_None_): A new input layer if you intend to fit the model on new data of a different size.
- **input_shape** (_None_): The size of images that the model is expected to take if you change the input layer.
- **pooling** (_None_): The [type of pooling](https://12ft.io/proxy?q=https%3A%2F%2Fmachinelearningmastery.com%2Fpooling-layers-for-convolutional-neural-networks%2F) to use when you are training a new set of output layers.
- **classes** (_1000_): The number of classes (e.g. size of output vector) for the model.

Next, let’s look at using the loaded VGG model to classify ad hoc photographs.

## Develop a Simple Photo Classifier

Let’s develop a simple image classification script.

### 1. Get a Sample Image

First, we need an image we can classify.

You can download a random photograph of a coffee mug from Flickr [here](https://12ft.io/proxy?q=https%3A%2F%2Fwww.flickr.com%2Fphotos%2Fjfanaian%2F4994221690%2F).

![Coffee Mug](data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20640%20480'%3E%3C/svg%3E)

Coffee Mug  
Photo by [jfanaian](https://12ft.io/proxy?q=https%3A%2F%2Fwww.flickr.com%2Fphotos%2Fjfanaian%2F4994221690%2F), some rights reserved.

Download the image and save it to your current working directory with the filename ‘_mug.jpg_‘.

### 2. Load the VGG Model

Load the weights for the VGG-16 model, as we did in the previous section.

|   |   |
|---|---|
|1<br><br>2<br><br>3|from keras.applications.vgg16 import VGG16<br><br># load the model<br><br>model=VGG16()|

### 3. Load and Prepare Image

Next, we can load the image as pixel data and prepare it to be presented to the network.

Keras provides some tools to help with this step.

First, we can use the _load_img()_ function to load the image and resize it to the required size of 224×224 pixels.

|   |   |
|---|---|
|1<br><br>2<br><br>3|from keras.preprocessing.image import load_img<br><br># load an image from file<br><br>image=load_img('mug.jpg',target_size=(224,224))|

Next, we can convert the pixels to a NumPy array so that we can work with it in Keras. We can use the _img_to_array()_ function for this.

|   |   |
|---|---|
|1<br><br>2<br><br>3|from keras.preprocessing.image import img_to_array<br><br># convert the image pixels to a numpy array<br><br>image=img_to_array(image)|

The network expects one or more images as input; that means the input array will need to be 4-dimensional: samples, rows, columns, and channels.

We only have one sample (one image). We can reshape the array by calling _reshape()_ and adding the extra dimension.

|   |   |
|---|---|
|1<br><br>2|# reshape data for the model<br><br>image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))|

Next, the image pixels need to be prepared in the same way as the ImageNet training data was prepared. Specifically, from the paper:

> The only preprocessing we do is subtracting the mean RGB value, computed on the training set, from each pixel.

— [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1409.1556), 2014.

Keras provides a function called _preprocess_input()_ to prepare new input for the network.

|   |   |
|---|---|
|1<br><br>2<br><br>3|from keras.applications.vgg16 import preprocess_input<br><br># prepare the image for the VGG model<br><br>image=preprocess_input(image)|

We are now ready to make a prediction for our loaded and prepared image.

### 4. Make a Prediction

We can call the _predict()_ function on the model in order to get a prediction of the probability of the image belonging to each of the 1000 known object types.

|   |   |
|---|---|
|1<br><br>2|# predict the probability across all output classes<br><br>yhat=model.predict(image)|

Nearly there, now we need to interpret the probabilities.

### 5. Interpret Prediction

Keras provides a function to interpret the probabilities called _decode_predictions()_.

It can return a list of classes and their probabilities in case you would like to present the top 3 objects that may be in the photo.

We will just report the first most likely object.

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7|from keras.applications.vgg16 import decode_predictions<br><br># convert the probabilities to class labels<br><br>label=decode_predictions(yhat)<br><br># retrieve the most likely result, e.g. highest probability<br><br>label=label[0][0]<br><br># print the classification<br><br>print('%s (%.2f%%)'%(label[1],label[2]*100))|

And that’s it.

### Complete Example

Tying all of this together, the complete example is listed below:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23|from keras.preprocessing.image import load_img<br><br>from keras.preprocessing.image import img_to_array<br><br>from keras.applications.vgg16 import preprocess_input<br><br>from keras.applications.vgg16 import decode_predictions<br><br>from keras.applications.vgg16 import VGG16<br><br># load the model<br><br>model=VGG16()<br><br># load an image from file<br><br>image=load_img('mug.jpg',target_size=(224,224))<br><br># convert the image pixels to a numpy array<br><br>image=img_to_array(image)<br><br># reshape data for the model<br><br>image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))<br><br># prepare the image for the VGG model<br><br>image=preprocess_input(image)<br><br># predict the probability across all output classes<br><br>yhat=model.predict(image)<br><br># convert the probabilities to class labels<br><br>label=decode_predictions(yhat)<br><br># retrieve the most likely result, e.g. highest probability<br><br>label=label[0][0]<br><br># print the classification<br><br>print('%s (%.2f%%)'%(label[1],label[2]*100))|

Running the example, we can see that the image is correctly classified as a “_coffee mug_” with a 75% likelihood.

|   |   |
|---|---|
|1|coffee_mug (75.27%)|

## Extensions

This section lists some ideas for extending the tutorial that you may wish to explore.

- **Create a Function**. Update the example and add a function that given an image filename and the loaded model will return the classification result.
- **Command Line Tool**. Update the example so that given an image filename on the command line, the program will report the classification for the image.
- **Report Multiple Classes**. Update the example to report the top 5 most likely classes for a given image and their probabilities.

## Further Reading

This section provides more resources on the topic if you are looking go deeper.

- [ImageNet](https://12ft.io/proxy?q=http%3A%2F%2Fwww.image-net.org%2F)
- [ImageNet on Wikipedia](https://12ft.io/proxy?q=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FImageNet)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://12ft.io/proxy?q=https%3A%2F%2Farxiv.org%2Fabs%2F1409.1556), 2015.
- [Very Deep Convolutional Networks for Large-Scale Visual Recognition](https://12ft.io/proxy?q=http%3A%2F%2Fwww.robots.ox.ac.uk%2F~vgg%2Fresearch%2Fvery_deep%2F), at Oxford.
- [Building powerful image classification models using very little data](https://12ft.io/proxy?q=https%3A%2F%2Fblog.keras.io%2Fbuilding-powerful-image-classification-models-using-very-little-data.html), 2016.
- [Keras Applications API](https://12ft.io/proxy?q=https%3A%2F%2Fkeras.io%2Fapplications%2F)
- [Keras weight files files](https://12ft.io/proxy?q=https%3A%2F%2Fgithub.com%2Ffchollet%2Fdeep-learning-models%2Freleases%2F)

## Summary

In this tutorial, you discovered the VGG convolutional neural network models for image classification.

Specifically, you learned:

- About the ImageNet dataset and competition and the VGG winning models.
- How to load the VGG model in Keras and summarize its structure.
- How to use the loaded VGG model to classifying objects in ad hoc photographs.

Do you have any questions?  
Ask your questions in the comments below and I will do my best to answer.
By [Bala Priya C](https://machinelearningmastery.com/author/bala-priya-c/ "Posts by Bala Priya C") on October 18, 2024 in [Machine Learning Resources](https://machinelearningmastery.com/category/machine-learning-resources/ "View all items in Machine Learning Resources") [4](https://machinelearningmastery.com/a-practical-guide-to-deploying-machine-learning-models/#comments)

 Share _Post_ Share

![A Practical Guide to Deploying Machine Learning Models](https://machinelearningmastery.com/wp-content/uploads/2024/10/mlm-deploy-ml-models.png)

Image by Author  
A Practical Guide to Deploying Machine Learning Models  

As a data scientist, you probably know how to build machine learning models. But it’s only when you deploy the model that you get a useful machine learning solution. And if you’re looking to learn more about deploying machine learning models, this guide is for you.

The steps involved in building and deploying ML models can typically be summed up like so: **building the model, creating an API to serve model predictions, containerizing the API, and deploying to the cloud**.

This guide focuses on the following:

- Building a machine learning model with Scikit-learn
- Creating a REST API to serve predictions from the model using FastAPI
- Containerizing the API using Docker

![deploy-ml-models](https://www.kdnuggets.com/wp-content/uploads/model-deployment.png)

Deploying ML Models | Image by Author  

We’ll build a simple regression model on the California housing dataset to predict house prices. By the end, you’ll have a containerized application that serves house price predictions based on selected input features.

## Setting Up the Project Environment

Before you start, make sure you have the following installed:

- A recent version of Python (Python 3.11 or later preferably)
- Docker for containerization; [Get Docker](https://docs.docker.com/get-started/get-docker/) for your operating system

⚙️ To follow along comfortably, it’s helpful to have a basic understanding of building machine learning models and working with APIs.

### Getting Started

Here’s the (recommended) structure for the project’s directory:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12|project-dir/<br><br>│<br><br>├── app/<br><br>│   ├── __init__.py       # Empty file<br><br>│   └── main.py           # FastAPI code for prediction API<br><br>│<br><br>├── model/<br><br>│   └── linear_regression_model.pkl  # Saved trained model (after running model_training.py)<br><br>│<br><br>├── model_training.py       # Script to train and save the model<br><br>├── requirements.txt       # Dependencies for the project<br><br>└── Dockerfile             # Docker configuration|

We’ll need a few Python libraries to get going. Let’s install them all next.

In your project environment, create and activate a virtual environment:

|   |   |
|---|---|
|1<br><br>2|$ python3 -m venv v1<br><br>$ source v1/bin/activate|

For the project we’ll be working on, we need pandas and scikit-learn to build the machine learning model. And FastAPI and Uvicorn to build the API to serve the model’s predictions.

So let’s install these required packages using pip:

|   |   |
|---|---|
|1|$ pip3 install pandas scikit-learn fastapi uvicorn|

You can find all the code for this tutorial [on GitHub](https://github.com/balapriyac/data-science-tutorials/tree/main/model_deployment).

## Building a Machine Learning Model

Now, we’ll train a linear regression model using the [California Housing dataset](https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.fetch_california_housing.html) which is built into scikit-learn. This model will predict house prices based on the selected features. In the project directory, create a file called **model_training.py**:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30<br><br>31<br><br>32<br><br>33|# model_training.py<br><br>import pandas as pd<br><br>from sklearn.datasets import fetch_california_housing<br><br>from sklearn.model_selection import train_test_split<br><br>from sklearn.linear_model import LinearRegression<br><br>import pickle<br><br>import os<br><br># Load the dataset<br><br>data = fetch_california_housing(as_frame=True)<br><br>df = data['data']<br><br>target = data['target']<br><br># Select a few features<br><br>selected_features = ['MedInc', 'AveRooms', 'AveOccup']<br><br>X = df[selected_features]<br><br>y = target<br><br># Train-test split<br><br>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)<br><br># Train the Linear Regression model<br><br>model = LinearRegression()<br><br>model.fit(X_train, y_train)<br><br># Create a 'model' folder to save the trained model<br><br>os.makedirs('model', exist_ok=True)<br><br># Save the trained model using pickle<br><br>with open('model/linear_regression_model.pkl', 'wb') as f:<br><br>pickle.dump(model, f)<br><br>print("Model trained and saved successfully.")|

This script loads the California housing dataset, selects three features (MedInc, AveRooms, AveOccup), trains a linear regression model, and saves it in the **model/** folder as **linear_regression_model.pkl**.

> **Note**: To keep things simple, we’ve only used a small subset of features. But you can try adding more.

Run the script to train the model and save it:

|   |   |
|---|---|
|1|$ python3 model_training.py|

You’ll get the following message and should be able to find the .pkl file in the **model/** directory:

|   |   |
|---|---|
|1|Model trained and saved successfully.|

## Creating the FastAPI App

We’ll now create an API that serves predictions using FastAPI.

Inside the **app/** folder, create two files: **__init__.py** (empty) and **main.py**. We do this because we’d like to [containerize the FastAPI app using Docker](https://fastapi.tiangolo.com/deployment/docker/) next.

In **main.py**, write the following code:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23<br><br>24<br><br>25<br><br>26<br><br>27<br><br>28<br><br>29<br><br>30|# app/main.py<br><br>from fastapi import FastAPI<br><br>from pydantic import BaseModel<br><br>import pickle<br><br>import os<br><br># Define the input data schema using Pydantic<br><br>class InputData(BaseModel):<br><br>    MedInc: float<br><br>    AveRooms: float<br><br>    AveOccup: float<br><br># Initialize FastAPI app<br><br>app = FastAPI(title="House Price Prediction API")<br><br># Load the model during startup<br><br>model_path = os.path.join("model", "linear_regression_model.pkl")<br><br>with open(model_path, 'rb') as f:<br><br>    model = pickle.load(f)<br><br>@app.post("/predict")<br><br>def predict(data: InputData):<br><br>    # Prepare the data for prediction<br><br>    input_features = [[data.MedInc, data.AveRooms, data.AveOccup]]<br><br>    # Make prediction using the loaded model<br><br>    prediction = model.predict(input_features)<br><br>    # Return the prediction result<br><br>    return {"predicted_house_price": prediction[0]}|

This FastAPI application exposes a **/predict** endpoint that takes three features (MedInc, AveRooms, AveOccup). It uses the trained model to predict house prices, and returns the predicted price.

## Containerizing the App with Docker

Now let’s containerize our FastAPI application. In the project’s root directory, create a **Dockerfile** and a **requirements.txt** file.

### Creating the Dockerfile

Let’s create a Dockerfile:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8<br><br>9<br><br>10<br><br>11<br><br>12<br><br>13<br><br>14<br><br>15<br><br>16<br><br>17<br><br>18<br><br>19<br><br>20<br><br>21<br><br>22<br><br>23|# Use Python 3.11 as the base image<br><br>FROM python:3.11-slim<br><br># Set the working directory inside the container<br><br>WORKDIR /code<br><br># Copy the requirements file<br><br>COPY ./requirements.txt /code/requirements.txt<br><br># Install the Python dependencies<br><br>RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt<br><br># Copy the app folder into the container<br><br>COPY ./app /code/app<br><br># Copy the model directory (with the saved model file) into the container<br><br>COPY ./model /code/model<br><br># Expose port 80 for FastAPI<br><br>EXPOSE 80<br><br># Command to run the FastAPI app with Uvicorn<br><br>CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]|

This creates a lightweight container for a FastAPI application using Python 3.11 (slim version) as the base image. It sets the working directory to **/code**, copies the **requirements.txt** file into the container, and installs the necessary Python dependencies without caching.

The FastAPI app and model files are then copied into the container. Port 80 is exposed for the application, and Uvicorn is used to run the FastAPI app. This makes the API accessible at port 80. This setup is efficient for deploying a FastAPI app in a containerized environment.

### Creating the requirements.txt File

Create a **requirements.txt** file listing all dependencies:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4|fastapi<br><br>uvicorn<br><br>scikit-learn<br><br>pandas|

## Building the Docker Image

Now that we have the Dockerfile, requirements.txt, and the FastAPI app ready, let’s build a Docker image and run the container.

![Dockerizing the API](https://www.kdnuggets.com/wp-content/uploads/containerize-app1.png)

Dockerizing the API | Image by Author  

Build the Docker image by running the following docker build command:

|   |   |
|---|---|
|1|$ docker build -t house-price-prediction-api .|

Next run the Docker container:

|   |   |
|---|---|
|1|$ docker run -d -p 80:80 house-price-prediction-api|

Your API should now be running and accessible at http://127.0.0.1:80.

You can use curl or Postman to test the **/predict** endpoint by sending a POST request. Here’s an example request:

|   |   |
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br>8|curl -X 'POST' \<br><br>  'http://127.0.0.1:80/predict' \<br><br>  -H 'Content-Type: application/json' \<br><br>  -d '{<br><br>  "MedInc": 3.5,<br><br>  "AveRooms": 5.0,<br><br>  "AveOccup": 2.0<br><br>}'|

This should return a response with the predicted house price, like this:

|   |   |
|---|---|
|1<br><br>2<br><br>3|{<br><br>  "predicted_house_price": 2.3248705765077062<br><br>}|

## Tagging and Pushing the Docker Image to Docker Hub

After building the Docker image, running the container, and testing it. You can now push it to Docker Hub for easier sharing and deploying to cloud platforms.

First, login to Docker Hub:

|   |   |
|---|---|
|1|$ docker login|

You’ll be prompted to enter the credentials.

Tag the Docker image:

|   |   |
|---|---|
|1|$ docker tag house-price-prediction-api your_username/house-price-prediction-api:v1|

Replace your_username with your Docker Hub username.

> **Note**: It also makes sense to add versions to your model files. When you update the model, you can rebuild the image with a new tag, and push the updated image to Docker Hub.

Push the image to Docker Hub:

|   |   |
|---|---|
|1|$ docker push your_username/house-price-prediction-api:v1|

Other developers can now pull and run the image like so:

|   |   |
|---|---|
|1<br><br>2|$ docker pull your_username/house-price-prediction-api:v1<br><br>$ docker run -d -p 80:80 your_username/house-price-prediction-api:v1|

Anyone with access to your Docker Hub repository can now pull the image and run the container.

## Wrap-up and Next Steps

Here’s a quick review of what we did in this tutorial:

- Train a machine learning model using scikit-learn
- Build a FastAPI application to serve predictions
- Containerize the application with Docker

We also looked at pushing the Docker image to Docker Hub for easier distribution. The next logical step is to deploy this containerized application to the cloud.

And for this, you can use services like AWS ECS, GCP, or Azure to deploy the API in a production environment. Let us know if you’d like a tutorial on deploying machine learning models to the cloud.

Happy deploying!

## References and Further Reading

- [FastAPI Tutorial: Build APIs with Python in Minutes](https://www.kdnuggets.com/fastapi-tutorial-build-apis-with-python-in-minutes)
- [Containerize Python Apps with Docker in 5 Easy Steps](https://www.kdnuggets.com/containerize-python-apps-with-docker-in-5-easy-steps)
- [FastAPI in Containers](https://fastapi.tiangolo.com/deployment/docker/)

 Share
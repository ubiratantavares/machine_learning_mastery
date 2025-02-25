import json

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping

import plotly.express as px
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from flask import Flask, request


server = Flask("mlm")
app = Dash(server=server,
           external_scripts=[
               "https://code.jquery.com/jquery-3.6.0.min.js"
           ])

# Load MNIST digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=3).astype("float32")
X_test = np.expand_dims(X_test, axis=3).astype("float32")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model_data = {
    "activation": "relu",
    "optimizer": "adam",
    "epochs": 100,
    "batchsize": 32,
    "model": None
}
train_status = {
    "running": False,
    "epoch": 0,
    "batch": 0,
    "batch metric": None,
    "last epoch": None,
}


class ProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        train_status["running"] = True
        train_status["epoch"] = 0
    def on_train_end(self, logs=None):
        train_status["running"] = False
    def on_epoch_begin(self, epoch, logs=None):
        train_status["epoch"] = epoch
        train_status["batch"] = 0
    def on_epoch_end(self, epoch, logs=None):
        train_status["last epoch"] = logs
    def on_train_batch_begin(self, batch, logs=None):
        train_status["batch"] = batch
    def on_train_batch_end(self, batch, logs=None):
        train_status["batch metric"] = logs


def train():
    activation = model_data["activation"]
    model = Sequential([
        Conv2D(6, (5, 5), activation=activation,
               input_shape=(28, 28, 1), padding="same"),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(16, (5, 5), activation=activation),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(120, (5, 5), activation=activation),
        Flatten(),
        Dense(84, activation=activation),
        Dense(10, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=model_data["optimizer"],
                  metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="val_loss", patience=3,
                              restore_best_weights=True)
    history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=model_data["epochs"],
            batch_size=model_data["batchsize"],
            verbose=0, callbacks=[earlystop, ProgressCallback()])
    return model, history


app.layout = html.Div(
    id="parent",
    children=[
        html.H1(
            children="LeNet5 training",
            style={"textAlign": "center"}
        ),
        html.Div(
            className="flex-container",
            children=[
                html.Div(children=[
                    html.Div(id="activationdisplay"),
                    dcc.Dropdown(
                        id="activation",
                        options=[
                            {"label": "Rectified linear unit", "value": "relu"},
                            {"label": "Hyperbolic tangent", "value": "tanh"},
                            {"label": "Sigmoidal", "value": "sigmoid"},
                        ],
                        value=model_data["activation"]
                    )
                ]),
                html.Div(children=[
                    html.Div(id="optimizerdisplay"),
                    dcc.Dropdown(
                        id="optimizer",
                        options=[
                            {"label": "Adam", "value": "adam"},
                            {"label": "Adagrad", "value": "adagrad"},
                            {"label": "Nadam", "value": "nadam"},
                            {"label": "Adadelta", "value": "adadelta"},
                            {"label": "Adamax", "value": "adamax"},
                            {"label": "RMSprop", "value": "rmsprop"},
                            {"label": "SGD", "value": "sgd"},
                            {"label": "FTRL", "value": "ftrl"},
                        ],
                        value=model_data["optimizer"]
                    ),
                ]),
                html.Div(children=[
                    html.Div(id="epochdisplay"),
                    dcc.Slider(1, 200, 1, marks={1: "1", 100: "100", 200: "200"},
                               value=model_data["epochs"], id="epochs"),
                ]),
                html.Div(children=[
                    html.Div(id="batchdisplay"),
                    dcc.Slider(1, 128, 1, marks={1: "1", 128: "128"},
                               value=model_data["batchsize"], id="batchsize"),
                ]),
            ]
        ),
        html.Button(id="train", n_clicks=0, children="Train"),
        html.Pre(id="progressdisplay"),
        dcc.Interval(id="trainprogress", n_intervals=0, interval=1000),
        dcc.Graph(id="historyplot"),
        html.Div(
            className="flex-container",
            id="predict",
            children=[
                html.Div(
                    children=html.Canvas(id="writing"),
                    style={"textAlign": "center"}
                ),
                html.Div(id="predictresult", children="?"),
                html.Pre(
                    id="lastinput",
                ),
            ]
        ),
        html.Div(id="dummy", style={"display": "none"}),
    ]
)


@app.callback(Output(component_id="epochdisplay", component_property="children"),
              Input(component_id="epochs", component_property="value"))
def update_epochs(value):
    model_data["epochs"] = value
    return f"Epochs: {value}"


@app.callback(Output("batchdisplay", "children"),
              Input("batchsize", "value"))
def update_batchsize(value):
    model_data["batchsize"] = value
    return f"Batch size: {value}"


@app.callback(Output("activationdisplay", "children"),
              Input("activation", "value"))
def update_activation(value):
    model_data["activation"] = value
    return f"Activation: {value}"


@app.callback(Output("optimizerdisplay", "children"),
              Input("optimizer", "value"))
def update_optimizer(value):
    model_data["optimizer"] = value
    return f"Optimizer: {value}"


@app.callback(Output("historyplot", "figure"),
              Input("train", "n_clicks"),
              State("activation", "value"),
              State("optimizer", "value"),
              State("epochs", "value"),
              State("batchsize", "value"),
              prevent_initial_call=True)
def train_action(n_clicks, activation, optimizer, epoch, batchsize):
    model_data.update({
        "activation": activation,
        "optimizer": optimizer,
        "epoch": epoch,
        "batchsize": batchsize,
    })
    model, history = train()
    model_data["model"] = model  # keep the trained model
    history = pd.DataFrame(history.history)
    fig = px.line(history, title="Model training metrics")
    fig.update_layout(xaxis_title="epochs",
                      yaxis_title="metric value", legend_title="metrics")
    return fig


@app.callback(Output("progressdisplay", "children"),
              Input("trainprogress", "n_intervals"))
def update_progress(n):
    return json.dumps(train_status, indent=4)


app.clientside_callback(
    "function() { pageinit(); };",
    Output("dummy", "children"),
    Input("dummy", "children")
)


@server.route("/recognize", methods=["POST"])
def recognize():
    if not model_data.get("model"):
        return "Please train your model."
    matrix = json.loads(request.form["matrix"])
    matrix = np.asarray(matrix).reshape(1, 28, 28)
    proba = model_data["model"].predict(matrix).reshape(-1)
    result = np.argmax(proba)
    return "Digit "+str(result)


# run server, with hot-reloading
app.run_server(debug=True, threaded=True)

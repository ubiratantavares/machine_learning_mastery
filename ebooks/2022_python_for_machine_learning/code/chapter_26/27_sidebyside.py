from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.layouts import row

# Load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Verify the shape of training data
total_examples, img_length, img_width = x_train.shape
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)


# Convert the dataset into a 2D array of shape 18623 x 784
x = convert_to_tensor(np.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=dtypes.float32)
# Eigen-decomposition from a 784 x 784 matrix
eigenvalues, eigenvectors = linalg.eigh(tensordot(transpose(x), x, axes=1))
# Print the three largest eigenvalues
print('3 largest eigenvalues: ', eigenvalues[-3:])
# Project the data to eigenvectors
x_pca = tensordot(x, eigenvectors, axes=1)


# Prepare for classifier network
epochs = 10
y_train = utils.to_categorical(train_labels)
input_dim = img_length*img_width
# Create a Sequential model
model = Sequential()
# First layer for reshaping input images from 2D to 1D
model.add(Reshape((input_dim, ), input_shape=(img_length, img_width)))
# Dense layer of 8 neurons
model.add(Dense(8, activation='relu'))
# Output layer
model.add(Dense(total_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33,
                    epochs=epochs, batch_size=10, verbose=0)


# Prepare pandas DataFrame
df_history = pd.DataFrame(history.history)
print(df_history)


# Create scatter plot in Bokeh
colormap = {0: "red", 1:"green", 2:"blue"}
my_scatter = figure(title="First Two Dimensions of Projected Data After Applying PCA",
                    x_axis_label="Dimension 1",
                    y_axis_label="Dimension 2",
                    width=500, height=400)
for digit in [0, 1, 2]:
    selection = x_pca[train_labels == digit]
    my_scatter.scatter(selection[:,-1].numpy(), selection[:,-2].numpy(),
                       color=colormap[digit], size=5, alpha=0.5,
                       legend_label="Digit "+str(digit))
my_scatter.legend.click_policy = "hide"


# Plot accuracy in Bokeh
p = figure(title="Training and validation accuracy",
           x_axis_label="Epochs", y_axis_label="Accuracy",
           width=500, height=400)
epochs_array = np.arange(epochs)
p.line(epochs_array, df_history['accuracy'], legend_label="Training",
       color="blue", line_width=2)
p.line(epochs_array, df_history['val_accuracy'], legend_label="Validation",
       color="green")
p.legend.click_policy = "hide"
p.legend.location = 'bottom_right'

show(row(my_scatter, p))

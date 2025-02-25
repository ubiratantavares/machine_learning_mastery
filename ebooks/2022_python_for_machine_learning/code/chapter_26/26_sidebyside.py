from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


# Plot side-by-side
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
# left plot
scatter = ax[0].scatter(x_pca[:, -1], x_pca[:, -2], c=train_labels, s=5)
legend_plt = ax[0].legend(*scatter.legend_elements(),
                         loc="lower left", title="Digits")
ax[0].add_artist(legend_plt)
ax[0].set_title('First Two Dimensions of Projected Data After Applying PCA')
# right plot
my_plot = sns.lineplot(data=df_history[["loss","val_loss"]], ax=ax[1])
my_plot.set_xlabel('Epochs')
my_plot.set_ylabel('Loss')
ax[1].legend(labels=["Training", "Validation"])
ax[1].set_title('Training and Validation Loss')
plt.show()

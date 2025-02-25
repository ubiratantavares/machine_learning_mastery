from tensorflow.keras.datasets import mnist
from tensorflow import dtypes, tensordot
from tensorflow import convert_to_tensor, linalg, transpose
import numpy as np
import matplotlib.pyplot as plt

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

# Create the plot
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=-60)
plt_3d = ax.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=train_labels, s=1)
plt.colorbar(plt_3d)
plt.show()

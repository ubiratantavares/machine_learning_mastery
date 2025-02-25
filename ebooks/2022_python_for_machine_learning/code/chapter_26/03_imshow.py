from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# load dataset
(x_train, train_labels), (_, _) = mnist.load_data()
# Choose only the digits 0, 1, 2
total_classes = 3
ind = np.where(train_labels < total_classes)
x_train, train_labels = x_train[ind], train_labels[ind]
# Shape of training data
total_examples, img_length, img_width = x_train.shape
# Print the statistics
print('Training data has ', total_examples, 'images')
print('Each image is of size ', img_length, 'x', img_width)

# Show images
img_per_row = 8
fig,ax = plt.subplots(nrows=2, ncols=img_per_row,
                      figsize=(18,4),
                      subplot_kw=dict(xticks=[], yticks=[]))
for row in [0, 1]:
    for col in range(img_per_row):
        ax[row, col].imshow(x_train[row*img_per_row + col].astype('int'))
plt.show()

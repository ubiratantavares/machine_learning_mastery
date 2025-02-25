from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
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

# Plot loss in seaborn
my_plot = sns.lineplot(data=df_history[["loss","val_loss"]])
my_plot.set_xlabel('Epochs')
my_plot.set_ylabel('Loss')
plt.legend(labels=["Training", "Validation"])
plt.title('Training and Validation Loss')
plt.show()

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(x_train, y_train), _ = keras.datasets.cifar10.load_data()
datagen = ImageDataGenerator()
data_iterator = datagen.flow(x_train, y_train, batch_size=8)

fig, ax = plt.subplots(nrows=4, ncols=8, figsize=(18,6),
                       subplot_kw=dict(xticks=[], yticks=[]))
for i in range(4):
    # The next() function will load 8 images from CIFAR
    X, Y = data_iterator.next()
    for j, img in enumerate(X):
        ax[i, j].imshow(img.astype('int'))
plt.show()

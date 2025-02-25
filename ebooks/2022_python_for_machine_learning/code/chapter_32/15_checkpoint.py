import os
from google.colab import drive
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model

MOUNTPOINT = "/content/gdrive"
DATADIR = os.path.join(MOUNTPOINT, "MyDrive")
drive.mount(MOUNTPOINT)

class LeNet5(tf.keras.Model):
  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = Conv2D(filters=6, kernel_size=(5,5), padding="same", activation="relu")
    self.max_pool2x2 = MaxPool2D(pool_size=(2,2))
    self.conv2 = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")
    self.flatten = Flatten()
    self.fc1 = Dense(units=120, activation="relu")
    self.fc2 = Dense(units=84, activation="relu")
    self.fc3=Dense(units=10, activation="softmax")
  def call(self, input_tensor):
    conv1 = self.conv1(input_tensor)
    maxpool1 = self.max_pool2x2(conv1)
    conv2 = self.conv2(maxpool1)
    maxpool2 = self.max_pool2x2(conv2)
    flatten = self.flatten(maxpool2)
    fc1 = self.fc1(flatten)
    fc2 = self.fc2(fc1)
    fc3 = self.fc3(fc2)
    return fc3

mnist_digits = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_digits.load_data()

# saving checkpoints
checkpoint_path = DATADIR + "/checkpoints/cp-epoch-{epoch}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
input_layer = Input(shape=(28,28,1))
model = LeNet5()(input_layer)
model = Model(inputs=input_layer, outputs=model)
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics="acc")
model.fit(x=train_images, y=train_labels, validation_data=[test_images, test_labels],
          batch_size=256, epochs=5, callbacks=[cp_callback])

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Model

class LeNet5(tf.keras.Model):
  def __init__(self):
    super(LeNet5, self).__init__()
    #creating layers in initializer
    self.conv1 = Conv2D(filters=6, kernel_size=(5,5), padding="same", activation="relu")
    self.max_pool2x2 = MaxPool2D(pool_size=(2,2))
    self.conv2 = Conv2D(filters=16, kernel_size=(5,5), padding="same", activation="relu")
    self.flatten = Flatten()
    self.fc1 = Dense(units=120, activation="relu")
    self.fc2 = Dense(units=84, activation="relu")
    self.fc3 = Dense(units=10, activation="softmax")
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

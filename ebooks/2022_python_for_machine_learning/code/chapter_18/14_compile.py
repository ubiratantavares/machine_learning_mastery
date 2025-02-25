from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(5, input_shape=(3,)),
    Dense(1)
])

has_loss = "loss" in dir(model)
print("Before compile, loss function defined:", has_loss)

model.compile()
has_loss = "loss" in dir(model)
print("After compile, loss function defined:", has_loss)

from tensorflow import keras

# Create model
model = keras.models.Sequential([
 	keras.layers.Input(shape=(10,)),
 	keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# using the .h5 extension in the file name specifies that the model
# should be saved in HDF5 format
model.save("my_model.h5")

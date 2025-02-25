import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the VGG16 model pre-trained on the ImageNet dataset
vgg16_model = vgg16.VGG16(weights='imagenet')

# Ask the user for manual inputs
image_path = input("Enter image path: ")
top_guesses = input("Enter number of top guesses: ")

# Load the image, resized according to the model target size
img_resized = image.load_img(image_path, target_size=(224, 224))

# Convert the image into an array
img = image.img_to_array(img_resized)

# Add in a dimension
img = np.expand_dims(img, axis=0)

# Scale the pixel intensity values
img = preprocess_input(img)

# Generate a prediction for the test image
pred_vgg = vgg16_model.predict(img)

# Decode and print the top 3 predictions
print('Prediction:', decode_predictions(pred_vgg, top=int(top_guesses)))

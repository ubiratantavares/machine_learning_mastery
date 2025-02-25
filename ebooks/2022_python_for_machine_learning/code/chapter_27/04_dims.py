from sklearn.datasets import load_digits
images = load_digits()["images"]
print(images.shape)

# image has axes 0, 1, and 2, adding axis 3
images = np.expand_dims(images, 3)
print(images.shape)

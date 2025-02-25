import sklearn.datasets

data = sklearn.datasets.fetch_california_housing(return_X_y=False, as_frame=True)
data = data["frame"]
print(data)

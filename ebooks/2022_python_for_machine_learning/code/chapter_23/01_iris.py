import sklearn.datasets

data, target = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)
data["target"] = target
print(data)

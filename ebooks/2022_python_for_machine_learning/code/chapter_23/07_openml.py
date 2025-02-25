import sklearn.datasets

data = sklearn.datasets.fetch_openml(data_id=42437, return_X_y=False, as_frame=True)
data = data["frame"]
print(data)

import sklearn.datasets

data = sklearn.datasets.fetch_openml("diabetes", version=1,
                                     as_frame=True, return_X_y=False)
data = data["frame"]
print(data)

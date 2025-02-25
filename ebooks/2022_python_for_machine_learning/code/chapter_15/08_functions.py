import numpy as np

DIST = "normal"

if DIST == "normal":
    rangen = np.random.normal
elif DIST == "uniform":
    rangen = np.random.uniform
else:
    raise NotImplementedError

random_data = rangen(size=(10,5))
print(random_data)

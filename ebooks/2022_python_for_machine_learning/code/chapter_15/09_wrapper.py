import numpy as np

DIST = "t"

if DIST == "normal":
    rangen = np.random.normal
elif DIST == "uniform":
    rangen = np.random.uniform
elif DIST == "t":
    def t_wrapper(size):
        # Student's t distribution with 3 degree of freedom
        return np.random.standard_t(df=3, size=size)
    rangen = t_wrapper
else:
    raise NotImplementedError

random_data = rangen(size=(10,5))
print(random_data)

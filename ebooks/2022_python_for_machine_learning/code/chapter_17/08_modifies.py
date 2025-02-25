import numpy as np
import pandas as pd

# function decorator to ensure numpy input
# and round off output to 4 decimal places
def ensure_numpy(fn):
    def decorated_function(data):
        array = np.asarray(data)
        output = fn(array)
        return np.around(output, 4)
    return decorated_function

@ensure_numpy
def numpysum(array):
    return array.sum()

x = np.random.randn(10,3)
y = pd.DataFrame(x, columns=["A", "B", "C"])

# output of numpy .sum() function
print("x.sum():", x.sum())
print()

# output of pandas .sum() funuction
print("y.sum():", y.sum())
print(y.sum())
print()

# calling decorated numpysum function
print("numpysum(x):", numpysum(x))
print("numpysum(y):", numpysum(y))

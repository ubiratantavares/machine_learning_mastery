import timeit
import numpy as np

time = timeit.timeit("np.random.random()", globals=globals())
print(time)

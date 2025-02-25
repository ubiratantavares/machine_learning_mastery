import functools
import numpy as np

power_2 = functools.partial(np.power, 2)
print('2^4 =', power_2(4))
print('2^6 =', power_2(6))

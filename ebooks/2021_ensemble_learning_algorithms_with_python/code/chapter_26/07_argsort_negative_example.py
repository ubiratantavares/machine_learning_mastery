# demonstrate argsort with negative scores
from numpy import argsort
# data
x = [-10, -100, -80]
print(x)
# argsort of data
print(argsort(x))
# arg sort of argsort of data
print(argsort(argsort(x)))
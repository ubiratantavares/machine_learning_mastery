# demonstrate argsort
from numpy import argsort
# data
x = [300, 100, 200]
print(x)
# argsort of data
print(argsort(x))
# arg sort of argsort of data
print(argsort(argsort(x)))
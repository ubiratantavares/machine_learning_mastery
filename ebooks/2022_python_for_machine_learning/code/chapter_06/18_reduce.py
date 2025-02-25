import functools
import operator

# Evaluates ((1+2)+3)+4
list_sum = functools.reduce(operator.add, [1, 2, 3, 4])
print(list_sum)

# Evaluates (2^3)^4
list_pow = functools.reduce(operator.pow, [2, 3, 4])
print(list_pow)

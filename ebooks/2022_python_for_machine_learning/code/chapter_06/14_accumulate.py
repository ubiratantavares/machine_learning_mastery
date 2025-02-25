import itertools
import operator

# Custom operator
def my_operator(a, b):
    return a+b if a>5 else a-b

x = [2, 3, 4, -6]
mul_result = itertools.accumulate(x, operator.mul)
print("After mul operator", list(mul_result))
pow_result = itertools.accumulate(x, operator.pow)
print("After pow operator", list(pow_result))
my_operator_result = itertools.accumulate(x, my_operator)
print("After customized my_operator", list(my_operator_result))

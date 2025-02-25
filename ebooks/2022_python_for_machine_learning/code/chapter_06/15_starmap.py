import itertools
import operator

pair_list = [(1, 2), (4, 0.5), (5, 7), (100, 10)]

starmap_add_result = itertools.starmap(operator.add, pair_list)
print("Starmap add result: ", list(starmap_add_result))

x1 = [2, 3, 4, -6]
x2 = [4, 3, 2, 1]

starmap_mul_result = itertools.starmap(operator.mul, zip(x1, x2))
print("Starmap mul result: ", list(starmap_mul_result))

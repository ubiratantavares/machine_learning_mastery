import functools
import operator

# All numbers from 1 to 20
input_list = list(range(20))
# Use map to see which numbers are divisible by 3
bool_list = map(lambda x: 1 if x%3==0 else 0, input_list)
# Convert map object to list
bool_list = list(bool_list)
print('bool_list =', bool_list)

total_divisible_3 = functools.reduce(operator.add, bool_list)
print('Total items divisible by 3 = ', total_divisible_3)

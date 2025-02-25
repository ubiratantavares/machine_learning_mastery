import itertools

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_result = itertools.filterfalse(lambda x: x%2, my_list)
small_terms = itertools.filterfalse(lambda x: x>=5, my_list)
print('Even result:', list(even_result))
print('Less than 5:', list(small_terms))

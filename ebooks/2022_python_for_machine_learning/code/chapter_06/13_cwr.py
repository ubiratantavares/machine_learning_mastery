import itertools

x = ['A', 'B', 'C']
for t in itertools.combinations_with_replacement(x, 2):
    print(t)

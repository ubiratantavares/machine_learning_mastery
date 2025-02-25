import itertools

x = ['A', 'B', 'C', 'D']
for t in itertools.combinations(x, 3):
    print(t)

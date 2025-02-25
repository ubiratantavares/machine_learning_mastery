import itertools

x = [1, 2, 3]
y = ['A', 'B']
for t in itertools.product(x, y):
    print(t)

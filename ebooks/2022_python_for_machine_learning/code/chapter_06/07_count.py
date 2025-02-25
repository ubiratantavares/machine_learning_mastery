import itertools

start = 0
step = 100
for i in itertools.count(start, step):
    print(i)
    if i >= 1000:
        break

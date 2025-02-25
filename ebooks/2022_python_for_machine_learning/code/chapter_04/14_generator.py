def get_fibonacci(x):
    x0 = 0
    x1 = 1
    for i in range(x):
        yield x0
        temp = x0 + x1
        x0 = x1
        x1 = temp

f = get_fibonacci(6)
for i in range(6):
    print(next(f))

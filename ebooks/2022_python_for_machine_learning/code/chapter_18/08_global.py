a = 1

def f(x):
    global a
    a = 2 * x
    return a

b = f(3)
print(a, b)

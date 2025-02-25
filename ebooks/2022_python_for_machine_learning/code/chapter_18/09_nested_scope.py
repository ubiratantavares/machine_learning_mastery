a = 1

def f(x):
    a = x
    def g(x):
        return a * x
    return g(3)

b = f(2)
print(b)

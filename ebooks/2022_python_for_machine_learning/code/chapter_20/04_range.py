def range(a, b=None, c=None):
    if c is None:
        c = 1
    if b is None:
        b = a
        a = 0
    values = []
    n = a
    while n < b:
        values.append(n)
        n = n + c
    return values

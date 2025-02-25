def add(a, b):
    try:
        a = float(a)
        b = float(b)
    except ValueError:
        raise ValueError("Input must be numbers")
    return a + b

add("one", "two")

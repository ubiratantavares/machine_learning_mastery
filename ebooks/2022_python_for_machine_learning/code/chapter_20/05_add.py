def add(a, b):
    assert isinstance(a, (int, float)), "`a` must be a number"
    assert isinstance(b, (int, float)), "`b` must be a number"
    return a + b

add("one", "two")

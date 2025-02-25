def isfloat(floatstring):
    if not isinstance(floatstring, str):
        raise ValueError("Expects a string input")
    seen_integer = False
    seen_dot = False
    seen_decimal = False
    for char in floatstring:
        if char.isdigit():
            if not seen_integer:
                seen_integer = True
            elif seen_dot and not seen_decimal:
                seen_decimal = True
        elif char == ".":
            if not seen_integer:
                return False  # e.g., ".3456"
            elif not seen_dot:
                seen_dot = True
            else:
                return False  # e.g., "1..23"
        else:
            return False  # e.g. "foo"
    if not seen_integer:
        return False   # e.g., ""
    if seen_dot and not seen_decimal:
        return False  # e.g., "2."
    return True


print(isfloat("foo"))       # False
print(isfloat(".3456"))     # False
print(isfloat("1.23"))      # True
print(isfloat("1..23"))     # False
print(isfloat("2"))         # True
print(isfloat("2."))        # False
print(isfloat("2,345.67"))  # False

def isfloat(floatstring):
    if not isinstance(floatstring, str):
        raise ValueError("Expects a string input")
    # States: "start", "integer", "dot", "decimal"
    state = "start"
    for char in floatstring:
        if state == "start":
            if char.isdigit():
                state = "integer"
            else:
                return False  # bad transition, can't continue
        elif state == "integer":
            if char.isdigit():
                pass  # stay in the same state
            elif char == ".":
                state = "dot"
            else:
                return False  # bad transition, can't continue
        elif state == "dot":
            if char.isdigit():
                state = "decimal"
            else:
                return False  # bad transition, can't continue
        elif state == "decimal":
            if not char.isdigit():
                return False  # bad transition, can't continue
    if state in ["integer", "decimal"]:
        return True
    else:
        return False

print(isfloat("foo"))       # False
print(isfloat(".3456"))     # False
print(isfloat("1.23"))      # True
print(isfloat("1..23"))     # False
print(isfloat("2"))         # True
print(isfloat("2."))        # False
print(isfloat("2,345.67"))  # False

def indentprint(x, indent=0, prefix="", suffix=""):
    if isinstance(x, dict):
        printdict(x, indent, prefix, suffix)
    elif isinstance(x, list):
        printlist(x, indent, prefix, suffix)
    elif isinstance(x, str):
        printstring(x, indent, prefix, suffix)
    else:
        printnumber(x, indent, prefix, suffix)

def printdict(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "{")
    for n, key in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(x[key], indent+2, str(key)+": ", comma)
    print(spaces + "}" + suffix)

def printlist(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + "[")
    for n, item in enumerate(x):
        comma = "," if n!=len(x)-1 else ""
        indentprint(item, indent+2, "", comma)
    print(spaces + "]" + suffix)

def printstring(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + '"' + str(x) + '"' + suffix)

def printnumber(x, indent, prefix, suffix):
    spaces = " " * indent
    print(spaces + prefix + str(x) + suffix)

data = {
    "a": [{
        "p": 3, "q": 4,
        "r": [3,4,5],
    },{
        "f": "foo", "g": 2.71
    },{
        "u": None, "v": "bar"
    }],
    "c": {
        "s": ["fizz", 2, 1.1],
        "t": []
    },
}

indentprint(data)

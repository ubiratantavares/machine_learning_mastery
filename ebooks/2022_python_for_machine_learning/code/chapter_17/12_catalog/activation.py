ACTIVATION = {}

def register(name):
    def decorator(fn):
        # assign fn to "name" key in ACTIVATION
        ACTIVATION[name] = fn
        # return fn unmodified
        return fn
    return decorator

def activate(x, kind):
    try:
        fn = ACTIVATION[kind]
        return fn(x)
    except KeyError:
        print("Activation function %s undefined" % kind)

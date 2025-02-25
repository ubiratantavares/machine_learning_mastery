import math

REGISTRY = {}

def register(name):
    def _decorator(fn):
        REGISTRY[name] = fn
        return fn
    return _decorator

@register("relu")
def rectified(x):
    return x if x > 0 else 0

@register("sigmoid")
def sigmoid(x):
    return 1/(1 + math.exp(-x))

def activate(x, funcname):
    if funcname not in REGISTRY:
        raise NotImplementedError(f"Function {funcname} is not implemented")
    else:
        func = REGISTRY[funcname]
        return func(x)

print(activate(1.23, "relu"))
print(activate(1.23, "sigmoid"))
print(activate(1.23, "tanh"))

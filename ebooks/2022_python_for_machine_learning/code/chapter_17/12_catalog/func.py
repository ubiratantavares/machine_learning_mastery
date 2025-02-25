from activation import register
import numpy as np

@register("relu")
def relu(x):
    return np.where(x>0, x, 0)

@register("sigmoid")
def sigm(x):
    return 1/(1+np.exp(-x))

@register("tanh")
def tanh(x):
    return np.tanh(x)

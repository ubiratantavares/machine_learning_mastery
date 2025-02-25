from numpy import array
from numpy import random
from numpy import dot
from scipy.special import softmax

# setting the value of the environment variable
import os
os.environ['PYTHONBREAKPOINT'] = ''

# defining our breakpoint() function
def breakpoint(*args, **kwargs):
    import importlib
    # reading the value of the environment variable
    val = os.environ.get('PYTHONBREAKPOINT')
    # if the value has been set to 0, skip all breakpoints
    if val == '0':
        return None
    # else if the value is an empty string, invoke the default pdb debugger
    elif len(val) == 0:
        hook_name = 'pdb.set_trace'
    # else, assign the value of the environment variable
    else:
        hook_name = val
    # split the string into the module name and the function name
    mod, dot, func = hook_name.rpartition('.')
    # get the function from the module
    module = importlib.import_module(mod)
    hook = getattr(module, func)

    return hook(*args, **kwargs)

# encoder representations of four different words
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

# stacking the word embeddings into a single array
words = array([word_1, word_2, word_3, word_4])

# generating the weight matrices
random.seed(42)
W_Q = random.randint(3, size=(3, 3))
W_K = random.randint(3, size=(3, 3))
W_V = random.randint(3, size=(3, 3))

# generating the queries, keys and values
Q = dot(words, W_Q)
K = dot(words, W_K)
V = dot(words, W_V)

# inserting a breakpoint
breakpoint()

# scoring the query vectors against all key vectors
scores = dot(Q, K.transpose())

# computing the weights by a softmax operation
weights = softmax(scores / K.shape[1] ** 0.5, axis=1)

# computing the attention by a weighted sum of the value vectors
attention = dot(weights, V)

print(attention)

import pickle

def test():
    return "Hello world!"

# Serialize and deserialize
pickled_function = pickle.dumps(test)
reconstructed_function = pickle.loads(pickled_function)

# Verify
print (reconstructed_function()) #prints "Hello, world!"

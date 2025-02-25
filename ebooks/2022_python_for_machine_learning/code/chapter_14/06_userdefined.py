import pickle

class NewClass:
    def __init__(self, data):
        print(data)
        self.data = data

# Create an object of NewClass
new_class = NewClass(1)

# Serialize and deserialize
pickled_data = pickle.dumps(new_class)
reconstructed = pickle.loads(pickled_data)

# Verify
print("Data from reconstructed object:", reconstructed.data)

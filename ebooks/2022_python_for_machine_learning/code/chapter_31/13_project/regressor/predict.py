import os
import pickle

def predict(features):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(current_dir, "model.pickle")
    with open(filepath, "rb") as fp:
        reg = pickle.load(fp)
    return reg.predict(features)

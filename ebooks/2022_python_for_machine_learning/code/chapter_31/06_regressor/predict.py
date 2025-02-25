import os
import pickle
import sys
import numpy as np

def predict(features):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(current_dir, "model.pickle")
    with open(filepath, "rb") as fp:
        reg = pickle.load(fp)
    return reg.predict(features)

if __name__ == "__main__":
    arr = np.asarray(sys.argv[1:]).astype(float).reshape(1,-1)
    y = predict(arr)
    print(y[0])

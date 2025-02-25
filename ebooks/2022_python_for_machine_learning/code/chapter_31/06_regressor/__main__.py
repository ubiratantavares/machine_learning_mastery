import sys
import numpy as np
from .predict import predict

if __name__ == "__main__":
    arr = np.asarray(sys.argv[1:]).astype(float).reshape(1,-1)
    y = predict(arr)
    print(y[0])

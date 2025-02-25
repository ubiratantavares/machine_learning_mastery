import numpy as np

X = np.array([
    [ 1.299,  0.332,  0.594, -0.047,  0.834],
    [ 0.842,  0.441, -0.705, -1.086, -0.252],
    [ 0.785,  0.478, -0.665, -0.532, -0.673],
    [ 0.062,  1.228, -0.333,  0.867,  0.371]
])

y = (X > 0).all(axis=0)
print(y)

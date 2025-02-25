import numpy as np
from regressor.predict import predict

X = np.asarray([[0.186,0,8.3,0,0.62,6.2,58,1.96,6,400,18.1,410,11.5]])
y = predict(X)
print(y[0])

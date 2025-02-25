import pandas as pd
import numpy as np

def datagen(X, y, batch_size, sampling_rate=0.7):
    """A generator to produce samples from input numpy arrays X and y
    """
    # Select rows from arrays X and y randomly
    indexing = np.random.random(len(X)) < sampling_rate
    Xsam, ysam = X[indexing], y[indexing]

    # Actual logic to generate batches
    def _gen(batch_size):
        while True:
            Xbatch, ybatch = [], []
            for _ in range(batch_size):
                i = np.random.randint(len(Xsam))
                Xbatch.append(Xsam[i])  # iloc?
                ybatch.append(ysam[i])  # iloc?
            yield np.array(Xbatch), np.array(ybatch)

    # Create and return a generator
    return _gen(batch_size)

X = pd.DataFrame(np.random.random((100,3)))
y = pd.DataFrame(np.random.random(100))

gen3 = datagen(X, y, 3)
print(next(gen3))

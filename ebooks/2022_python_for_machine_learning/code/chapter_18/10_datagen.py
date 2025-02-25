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
                Xbatch.append(Xsam[i])
                ybatch.append(ysam[i])
            yield np.array(Xbatch), np.array(ybatch)

    # Create and return a generator
    return _gen(batch_size)

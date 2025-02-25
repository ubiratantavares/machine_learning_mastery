from typing import List, Tuple, Generator
import pandas as pd
import numpy as np

TrainingSampleGenerator = Generator[Tuple[np.ndarray,np.ndarray], None, None]

def lstm_gen(data: pd.DataFrame,
             timesteps: int,
             batch_size: int) -> TrainingSampleGenerator:
    """Generator to produce random samples for LSTM training

    Args:
        data: DataFrame of data with datetime index in chronological order,
              samples are drawn from this
        timesteps: Number of time steps for each sample, data will be
                   produced from a window of such length
        batch_size: Number of samples in each batch

    Yields:
        ndarray, ndarray: The (X,Y) training samples drawn on a random window
        from the input data
    """
    input_columns = [c for c in data.columns if c != "target"]
    batch: List[Tuple[pd.DataFrame, pd.Series]] = []
    while True:
        # pick one start time and security
        while True:
            # Start from a random point from the data and clip a window
            row = data["target"].sample()
            starttime = row.index[0]
            window: pd.DataFrame = data[starttime:].iloc[:timesteps]
            # If we are at the end of the DataFrame, we can't get a full
            # window and we must start over
            if len(window) == timesteps:
                break
        # Extract the input and output
        y = window["target"]
        X = window[input_columns]
        batch.append((X, y))
        # If accumulated enough for one batch, dispatch
        if len(batch) == batch_size:
            X, y = zip(*batch)
            yield np.array(X).astype("float32"), np.array(y).astype("float32")
            batch = []

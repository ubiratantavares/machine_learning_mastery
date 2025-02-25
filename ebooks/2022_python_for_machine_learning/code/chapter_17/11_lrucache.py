import functools

import pandas_datareader as pdr

# memoize using lru_cache
@functools.lru_cache
def get_stock_data(ticker):
    # pull data from stooq
    df = pdr.stooq.StooqDailyReader(symbols=ticker, start="1/1/00", end="31/12/21"
                                   ).read()
    return df

# testing call to function
import cProfile as profile
import pstats

for i in range(1, 3):
    print(f"Run {i}")
    run_profile = profile.Profile()
    run_profile.enable()
    get_stock_data("^DJI")
    run_profile.disable()
    pstats.Stats(run_profile).print_stats(0)

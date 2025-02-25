import pickle
import hashlib
import pandas_datareader as pdr


MEMO = {} # To remember the function input and output

def memoize(fn):
    def _deco(*args, **kwargs):
        # pickle the function arguments and obtain hash as the store keys
        key = (fn.__name__, hashlib.md5(pickle.dumps((args, kwargs), 4)).hexdigest())
        # check if the key exists
        if key in MEMO:
            ret = pickle.loads(MEMO[key])
        else:
            ret = fn(*args, **kwargs)
            MEMO[key] = pickle.dumps(ret)
        return ret
    return _deco


@memoize
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

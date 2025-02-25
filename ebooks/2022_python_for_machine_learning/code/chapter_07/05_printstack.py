import traceback
import random

def compute():
    n = random.randint(0, 10)
    m = random.randint(0, 10)
    return n/m

def compute_many(n_times):
    try:
        for _ in range(n_times):
            x = compute()
        print(f"Completed {n_times} times")
    except:
        print("Something wrong")
        traceback.print_exc()

compute_many(100)

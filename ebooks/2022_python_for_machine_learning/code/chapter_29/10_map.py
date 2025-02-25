import multiprocessing
import time

def cube(x):
    return x**3

if __name__ == "__main__":
    pool = multiprocessing.Pool(3)
    start_time = time.perf_counter()
    result = pool.map(cube, range(1,1000))
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)

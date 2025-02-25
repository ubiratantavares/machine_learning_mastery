import multiprocessing

def cube(x):
    return x**3

if __name__ == "__main__":
    # this does not work
    processes = [multiprocessing.Process(target=cube, args=(x,)) for x in range(1,1000)]
    [p.start() for p in processes]
    result = [p.join() for p in processes]
    print(result)

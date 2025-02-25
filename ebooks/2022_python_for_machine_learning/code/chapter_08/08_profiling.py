import timeit
measurements = timeit.repeat('[x**0.5 for x in range(1000)]', number=10000)
print(measurements)

import functools

@functools.lru_cache()
def fib(n):
    global count
    count = count + 1
    return fib(n-2) + fib(n-1) if n>1 else 1

def fib_slow(n):
    global slow_count
    slow_count = slow_count + 1
    return fib_slow(n-2) + fib_slow(n-1) if n>1 else 1

count = 0
slow_count = 0
fib(30)
fib_slow(30)

print('With lru_cache total function evaluations: ', count)
print('Without lru_cache total function evaluations: ', slow_count)

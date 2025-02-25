import math

def square_decorator(function):
    def square_it(arg):
        x = function(arg)
        return x*x
    return square_it

size_sq = square_decorator(len)
print(size_sq([1,2,3]))

sin_sq = square_decorator(math.sin)
print(sin_sq(math.pi/4))

@square_decorator
def plus_one(a):
    return a+1

a = plus_one(3)
print(a)

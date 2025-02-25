from math import sqrt

def quadratic(a,b,c):
    discrim = b*b - 4*a*c
    x = -b/(2*a)
    y = sqrt(discrim)/(2*a)
    return x-y, x+y

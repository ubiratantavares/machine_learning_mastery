a = 42
b = "foo"
print("a is %s; b is %s" % (a,b))
a, b = b, a # swap
print("After swap, a is %s; b is %s" % (a,b))

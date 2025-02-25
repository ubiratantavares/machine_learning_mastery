A = [1, 2, "fizz", 4, "buzz", "fizz", 7]
A += [8, "fizz", "buzz", 11, "fizz", 13, 14, "fizzbuzz"]
print(A)
A[2:2] = [2.1, 2.2]
print(A)
A[0:2] = []
print(A)

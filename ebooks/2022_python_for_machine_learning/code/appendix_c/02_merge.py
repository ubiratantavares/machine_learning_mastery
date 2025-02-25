A = {'one': 1, 'two': 2}
B = {'three': 3, 'four': 4}
C = {**A, **B}

print(C)
# {'one': 1, 'two': 2, 'three': 3, 'four': 4}

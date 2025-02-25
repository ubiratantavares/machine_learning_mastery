A = {'one': 1, 'two': 2}
B = {'three': 3, 'four': 4}
C = dict((k,v) for k,v in list(A.items())+list(B.items()))

print(C)
# {'one': 1, 'two': 2, 'three': 3, 'four': 4}

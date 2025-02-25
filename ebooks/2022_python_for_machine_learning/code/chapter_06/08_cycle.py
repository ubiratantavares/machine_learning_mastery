import itertools

counter = 0
cyclic_list = [1, 2, 3, 4, 5]

for i in itertools.cycle(cyclic_list):
    print(i)
    counter = counter+1
    if counter>10:
        break

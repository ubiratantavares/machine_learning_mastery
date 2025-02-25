sentence = "Portez ce vieux whisky au juge blond qui fume"
counter = {}
for char in sentence:
    if char not in counter:
        counter[char] = 0
    counter[char] += 1

print(counter)

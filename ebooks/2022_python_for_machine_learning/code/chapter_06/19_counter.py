import functools

def addcount(counter, element):
    if element not in counter:
        counter[element] = 1
    else:
        counter[element] += 1
    return counter

items = ["a", "b", "a", "c", "d", "c", "b", "a"]

counts = functools.reduce(addcount, items, {})
print(counts)

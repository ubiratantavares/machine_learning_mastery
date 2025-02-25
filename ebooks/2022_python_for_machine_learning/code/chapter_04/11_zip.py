name = ['Triangle', 'Square', 'Hexagon', 'Pentagon']
sides = [3, 4, 6, 5]
colors = ['red', 'green', 'yellow', 'blue']
shapes = zip(name, sides, colors)

# Tuples are created from one item from each list
print(set(shapes))

# Easy to use enumerate and zip together for iterating through multiple lists in one go
for i, (n, s, c) in enumerate(zip(name, sides, colors)):
    print(i, 'Shape- ', n, '; Sides ', s)

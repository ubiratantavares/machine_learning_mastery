colors = ["red", "green", "blue"]
animals = ["cat", "dog", "bird"]
newlist = []
for c in colors:
    for a in animals:
        newlist.append(c + " " + a)
print(newlist)

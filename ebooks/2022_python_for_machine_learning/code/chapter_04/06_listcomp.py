colors = ["red", "green", "blue"]
animals = ["cat", "dog", "bird"]

newlist = [c+" "+a for c in colors for a in animals]
print(newlist)

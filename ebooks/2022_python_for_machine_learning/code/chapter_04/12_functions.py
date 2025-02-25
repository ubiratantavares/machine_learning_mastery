import math

def circle(r):
    area = 0
    def area_obj():
        nonlocal area
        area = math.pi * r * r
        print("area_obj")
    return area_obj

def circle(r):
    area_val = math.pi * r * r
    def area():
        print(area_val)
    return area

# returns area_obj as a function. The value of r passed is retained
circle_1 = circle(1)
circle_2 = circle(2)

# Calling area_obj() with radius = 1
circle_1()
# Calling area_obj() with radius = 2
circle_2()

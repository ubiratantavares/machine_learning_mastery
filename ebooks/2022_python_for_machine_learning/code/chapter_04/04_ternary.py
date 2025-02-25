original_list = [1, 2, 3, 4]
new_list = [i if i%2==0 else 0 for i in original_list]
print(new_list)

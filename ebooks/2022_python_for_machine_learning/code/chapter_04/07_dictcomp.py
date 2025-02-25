original_dict = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
new_dict = {key:'number ' + value for (key, value) in original_dict.items()}
print(new_dict)

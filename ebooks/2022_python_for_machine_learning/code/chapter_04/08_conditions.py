original_dict = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}

# Only add keys which are greater than 2
new_dict_high_keys = {key:'number ' + value
                      for (key, value) in original_dict.items() if key>2}
print(new_dict_high_keys)

# Only change values with key>2
new_dict_2 = {key:('number ' + value if key>2 else value)
              for (key, value) in original_dict.items() }
print(new_dict_2)

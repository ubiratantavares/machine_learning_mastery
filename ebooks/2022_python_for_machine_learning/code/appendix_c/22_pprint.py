import json
import pprint

data = {'a': 1, 'b': [2.1, 3.2, 4.3], 'c':{'d': 5}, 'e':{'f': 'foo', 'g':'bar', 'h':{"foobar": None}}}
print(data)
print(json.dumps(data, indent=4))
pprint.pprint(data)
pprint.pprint(data, indent=4, depth=2)

import json
import h5py

with h5py.File("my_model.h5", "r") as infile:
    for key in infile.attrs.keys():
        formatted = infile.attrs[key]
        if key.endswith("_config"):
            formatted = json.dumps(json.loads(formatted), indent=4)
        print(f"{key}: {formatted}")

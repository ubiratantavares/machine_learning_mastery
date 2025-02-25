import h5py

with h5py.File("test.hdf5", "r") as file:
    print (file.keys()) #gets names of datasets that are in the file
    dataset = file["test_dataset"]

import h5py

with h5py.File("test.hdf5", "w") as file:
    # creates dataset inside group1
    file.create_dataset("group1/dataset1", shape=(10,))

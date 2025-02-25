import h5py

with h5py.File("test.hdf5", "w") as file:
    # creates new group_1 in file
    file.create_group("group_1")
    group1 = file["group_1"]
    # creates dataset inside group1
    group1.create_dataset("dataset1", shape=(10,))
    # to access the dataset
    dataset = file["group_1"]["dataset1"]

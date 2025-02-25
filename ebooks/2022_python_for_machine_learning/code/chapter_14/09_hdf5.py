import h5py

with h5py.File("test.hdf5", "w") as file:
    dataset = file.create_dataset("test_dataset", (100,), dtype="i4")

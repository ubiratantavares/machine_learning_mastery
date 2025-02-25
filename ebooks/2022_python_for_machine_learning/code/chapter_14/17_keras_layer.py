import h5py

with h5py.File("my_model.h5", "r") as infile:
    print(infile["/model_weights/dense/dense/kernel:0"][:])

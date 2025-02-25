import h5py # type: ignore


def dumphdf5(filename: str) -> int:
    """Open a HDF5 file and print all the dataset and attributes stored

    Args:
        filename: The HDF5 filename

    Returns:
        Number of dataset found in the HDF5 file
    """
    count: int = 0

    def recur_dump(obj) -> None:
        nonlocal count
        print(f"{obj.name} ({type(obj).__name__})")
        if obj.attrs.keys():
            print("\tAttribs:")
            for key in obj.attrs.keys():
                print(f"\t\t{key}: {obj.attrs[key]}")
        if isinstance(obj, h5py.Group):
            # Group has key-value pairs
            for key, value in obj.items():
                recur_dump(value)
        elif isinstance(obj, h5py.Dataset):
            count += 1
            print(obj[()])

    with h5py.File(filename) as obj:
        recur_dump(obj)
        print(f"{count} dataset found")
    return count


dumphdf5("my_model.h5")

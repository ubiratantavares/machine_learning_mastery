def binary_search(array, target):
    """Binary search on array for target

    Args:
        array: sorted array
        target: the element to search for
    Returns:
        index n on the array such that array[n]==target
        if the target not found, return -1
    """
    s,e = 0, len(array)
    while s < e:
        m = (s+e)//2
        if array[m] == target:
            return m
        elif array[m] > target:
            e = m
        elif array[m] < target:
            s = m+1
        assert m != (s+e)//2, "we didn't move our midpoint"
    return -1

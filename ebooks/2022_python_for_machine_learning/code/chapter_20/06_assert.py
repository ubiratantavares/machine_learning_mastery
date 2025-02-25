def evenitems(arr):
    newarr = []
    for i in range(len(arr)):
        if i % 2 == 0:
            newarr.append(arr[i])
    assert len(newarr) * 2 >= len(arr)
    return newarr

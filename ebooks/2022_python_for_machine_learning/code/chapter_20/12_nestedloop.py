def neg_in_upper_tri(matrix):
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    for i in range(n_rows):
        for j in range(n_cols):
            if i > j:
                continue  # we are not in upper triangular
            if matrix[i][j] < 0:
                return True
    return False

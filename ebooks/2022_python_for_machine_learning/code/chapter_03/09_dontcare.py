import pandas as pd
A = pd.DataFrame([[11,12,13],[12,13,14],[13,14,15],[15,16,17]], columns=["x","y","z"])
print(A)

for _, row in A.iterrows():
    print(row["z"])

import sqlite3

import pandas as pd
from sklearn.datasets import fetch_openml

# Read dataset from OpenML
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
print("Data from OpenML:")
print(type(dataset))
print(dataset.head())

# Create database
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
create_sql = """
    CREATE TABLE diabetes(
        preg NUM,
        plas NUM,
        pres NUM,
        skin NUM,
        insu NUM,
        mass NUM,
        pedi NUM,
        age NUM,
        class TEXT
    )
"""
cur.execute(create_sql)

# Insert data into the table using a parameterized SQL
insert_sql = "INSERT INTO diabetes VALUES (?,?,?,?,?,?,?,?,?)"
rows = dataset.to_numpy().tolist()
cur.executemany(insert_sql, rows)

def cursor2dataframe(cur):
    """Read the column header from the cursor and then the rows of
    data from it. Afterwards, create a DataFrame"""
    header = [x[0] for x in cur.description]
    # gets data from the last executed SQL query
    data = cur.fetchall()
    # convert the data into a pandas DataFrame
    return pd.DataFrame(data, columns=header)

# get 5 random rows from the diabetes table
select_sql = "SELECT * FROM diabetes ORDER BY random() LIMIT 5"
cur.execute(select_sql)
sample = cursor2dataframe(cur)
print("Data from SQLite database:")
print(sample)

# close database connection
conn.commit()
conn.close()

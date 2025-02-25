import sqlite3

import numpy as np
from sklearn.datasets import fetch_openml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create database
conn = sqlite3.connect("diabetes.db")
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS diabetes")
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

# Read data from OpenML, insert data into the table using a parameterized SQL
dataset = fetch_openml("diabetes", version=1, as_frame=True, return_X_y=False)["frame"]
insert_sql = "INSERT INTO diabetes VALUES (?,?,?,?,?,?,?,?,?)"
rows = dataset.to_numpy().tolist()
cur.executemany(insert_sql, rows)

# Commit to flush change to disk, then close connection
conn.commit()
conn.close()

# Create data generator for Keras classifier model
def datagen(batch_size):
    """A generator to produce samples from database
    """
    # Tensorflow may run in different thread, thus needs check_same_thread=False
    conn = sqlite3.connect("diabetes.db", check_same_thread=False)
    cur = conn.cursor()
    sql = f"""
        SELECT preg, plas, pres, skin, insu, mass, pedi, age, class
        FROM diabetes
        ORDER BY random()
        LIMIT {batch_size}
    """
    while True:
        # Read rows from database
        cur.execute(sql)
        data = cur.fetchall()
        # Extract features
        X = [row[:-1] for row in data]
        # Extract targets, encode into binary (0 or 1)
        y = [1 if row[-1]=="tested_positive" else 0 for row in data]
        yield np.asarray(X), np.asarray(y)

# create binary classification model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(datagen(32), epochs=5, steps_per_epoch=2000)

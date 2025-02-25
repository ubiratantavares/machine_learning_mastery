import dbm
import pickle
import random

import numpy as np
import sklearn.datasets

# get digits dataset (8x8 images of digits)
digits = sklearn.datasets.load_digits()

# create file if not exists, otherwise open for read/write
with dbm.open("digits.dbm", "c") as db:
    for idx in range(len(digits.target)):
        db[str(idx)] = pickle.dumps((digits.images[idx], digits.target[idx]))

# number of images that we want in our sample
batchsize = 4
images = []
targets = []

# open the database and read a sample
with dbm.open("digits.dbm", "r") as db:
    # get all keys from the database
    keys = db.keys()
    # randomly samples n keys
    for key in random.sample(keys, batchsize):
        # go through each key in the random sample
        image, target = pickle.loads(db[key])
        images.append(image)
        targets.append(target)
    print(np.array(images), np.array(targets))

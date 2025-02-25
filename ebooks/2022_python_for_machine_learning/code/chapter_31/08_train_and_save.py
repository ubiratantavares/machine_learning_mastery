import pickle
from regressor.train import train

model = train()
with open("model.pickle", "wb") as fp:
    pickle.dump(model, fp)

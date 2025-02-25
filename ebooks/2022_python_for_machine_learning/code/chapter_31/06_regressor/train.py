import os
import json
import pickle
from sklearn.linear_model import LinearRegression

def load_data():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(current_dir, "data.json")
    data = json.load(open(filepath))
    return data

def train():
    reg = LinearRegression()
    data = load_data()
    reg.fit(data["data"], data["target"])
    return reg

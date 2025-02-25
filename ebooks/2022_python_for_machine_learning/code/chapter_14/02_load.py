import pickle

with open("test.pickle", "rb") as infile:
 	test_dict_reconstructed = pickle.load(infile)

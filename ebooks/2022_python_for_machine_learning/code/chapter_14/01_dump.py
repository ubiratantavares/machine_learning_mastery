import pickle

test_dict = {"Hello": "World!"}
with open("test.pickle", "wb") as outfile:
 	# "wb" argument opens the file in binary mode
	pickle.dump(test_dict, outfile)

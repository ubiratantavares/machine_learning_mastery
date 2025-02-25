class Dog:
	family = "Canine"

	def __init__(self, name, breed):
		self.name = name
		self.breed = breed
		self.tricks = []

	def add_tricks(self, x):
		self.tricks.append(x)

	def info(self, x):
		self.add_tricks(x)
		print(self.name, "is a female", self.breed, "that", self.tricks[0])

dog1 = Dog("Lassie", "Rough Collie")
dog1.info("barks on command")

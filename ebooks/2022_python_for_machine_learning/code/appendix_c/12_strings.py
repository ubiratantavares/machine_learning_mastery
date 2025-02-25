fruit = "apples"
count = 5

# All these are the same
sentence = f"I have {count} {fruit}"
sentence = "I have {} {}".format(count, fruit)
sentence = "I have {:d} {:s}".format(count, fruit)
sentence = "I have {c:d} {f:s}".format(c=count, f=fruit)
sentence = "I have %d %s" % (count, fruit)
sentence = "I have %(c)d %(f)s" % {"c":count, "f":fruit}

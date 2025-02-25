import sys
import pdb
import random

def debughook(etype, value, tb):
    pdb.pm() # post-mortem debugger
sys.excepthook = debughook

# Experimentally find the average of 1/x where x is a random integer in 0 to 9999
N = 1000
randomsum = 0
for i in range(N):
    x = random.randint(0,10000)
    randomsum += 1/x

print("Average is", randomsum/N)

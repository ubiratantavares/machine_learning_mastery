from scipy.stats import norm
n = norm.cdf([1,2,3,-1,-2,-3])
print(n)
print(n[:3] - n[-3:])

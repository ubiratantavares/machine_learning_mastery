"""
Continuous Probability Distributions for Machine Learning
https://machinelearningmastery.com/continuous-probability-distributions-for-machine-learning/
"""

# sample a normal distribution
from numpy.random import normal

# pdf and cdf for a normal distribution
from scipy.stats import norm

# sample an exponential distribution
from numpy.random import exponential

# pdf and cdf for an exponential distribution
from scipy.stats import expon

# sample a pareto distribution
from numpy.random import pareto as p1

# pdf and cdf for a pareto distribution
from scipy.stats import pareto

from matplotlib import pyplot as plt

"""
Normal Distribution
"""

# define the distribution
mu = 50
sigma = 5
n = 10

# generate the sample
sample = normal(mu, sigma, n)
print(sample)

# define distribution parameters
mu = 50
sigma = 5

# create distribution
dist = norm(mu, sigma)

# calculate the values that define the middle 95%
p = 0.95

limit_inf = (1 - p)/2.0
limit_sup = p + limit_inf

low_end = dist.ppf(limit_inf)
high_end = dist.ppf(limit_sup)

print('Middle 95% between {:.1f} and {:.1f}'.format(low_end, high_end))

# plot pdf
values = [value for value in range(1, 101)]

probabilities = [dist.pdf(value) for value in values]

plt.plot(values, probabilities)
plt.axvline(low_end, color='r')
plt.axvline(high_end, color='r')
plt.grid()
plt.show()

# plot cdf
cprobs = [dist.cdf(value) for value in values]
plt.plot(values, cprobs)
plt.grid()
plt.show()


"""
Exponential Distribution
"""


# define the distribution
beta = 50
n = 10

# generate the sample
sample = exponential(beta, n)
print(sample)


# define distribution parameter
beta = 50

# create distribution
dist = expon(beta)

# plot pdf
values = [value for value in range(50, 100)]

probabilities = [dist.pdf(value) for value in values]
plt.plot(values, probabilities)
plt.grid()
plt.show()

# plot cdf
cprobs = [dist.cdf(value) for value in values]
plt.plot(values, cprobs)
plt.grid()
plt.show()


"""
Pareto Distribution
"""


# define the distribution
alpha = 1.1
n = 10

# generate the sample
sample = p1(alpha, n)
print(sample)

# define distribution parameter
alpha = 1.5

# create distribution
dist = pareto(alpha)

# plot pdf
values = [value for value in range(10, 100)]
probabilities = [dist.pdf(value) for value in values]

plt.plot(values, probabilities)
plt.grid()
plt.show()

# plot cdf
cprobs = [dist.cdf(value) for value in values]

plt.plot(values, cprobs)
plt.grid()
plt.show()

"""
Database:

    Não utilizado.

Tutorials:

    Não utilizado.

Wikipedia:

    Probability density function
        https://en.wikipedia.org/wiki/Probability_density_function
        
    Exponential family
        https://en.wikipedia.org/wiki/Exponential_family

    Normal Distribution 
        https://en.wikipedia.org/wiki/Normal_distribution
        
    Log-normal distribution
        https://en.wikipedia.org/wiki/Log-normal_distribution
        
    68–95–99.7 rule    
        https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
        
    Exponential Distribution
        https://en.wikipedia.org/wiki/
    
    Geometric distribution
        https://en.wikipedia.org/wiki/Geometric_distribution
        
    Laplace distribution
        https://en.wikipedia.org/wiki/Laplace_distribution
        
    Pareto Distribution    
        https://en.wikipedia.org/wiki/Pareto_distribution
        
    Pareto principle
        https://en.wikipedia.org/wiki/Pareto_principle
        
    Power law
        https://en.wikipedia.org/wiki/Power_law
        
Books

    Chapter 2: Probability Distributions
        Pattern Recognition and Machine Learning, 2006, by Christopher M. Bishop (Author)

    Section 3.9: Common Probability Distributions
        Deep Learning, 2016, by Ian Goodfellow  (Author), Yoshua Bengio  (Author), Aaron Courville  (Author)

    Section 2.3: Some common discrete distributions
        Machine Learning: A Probabilistic Perspective, 2012, by Kevin P. Murphy  (Author)


API:
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html
"""
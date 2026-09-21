from random import uniform
from math import log

def gaussian(mu, sigma, n):
    g = []
    while len(g) < n:
        u1 = uniform(-1, 1)
        u2 = uniform(-1, 1)
        s = u1 * u1 + u2 * u2
        if s > 0 and s < 1:
            k = (-2 * log(s) / s) ** 0.5
            x = u1 * k
            y = u2 * k
            g.append(mu + sigma * x)
            if len(g) < n:
                g.append(mu + sigma * y)
    return g

# Mean values
mu1 = 5
mu2 = 10

# Variance = 1
# Therefore standard deviation = sqrt(1) = 1
sigma = 1

# Number of samples
n = 10000

# Generate two Gaussian datasets
D1 = gaussian(mu1, sigma, n)
D2 = gaussian(mu2, sigma, n)

print("D1:")
print(D1)

print("\nD2:")
print(D2)
from gaussian import D1, D2

# Mean
mu1 = 5
mu2 = 10

# Changed covariance
# C = [ sigma11^2     0     ]
#     [     0     sigma22^2 ]

sigma11 = 2
sigma22 = 1

n = len(D1)

# Y = mu + C^(1/2)(X - mu)

Y1 = []
Y2 = []

for i in range(n):
    y1 = mu1 + sigma11 * (D1[i] - mu1)
    y2 = mu2 + sigma22 * (D2[i] - mu2)

    Y1.append(y1)
    Y2.append(y2)

# Combine Y into 2-D data
data = []

for i in range(n):
    data.append([Y1[i], Y2[i]])

# Mean vector
mu = [mu1, mu2]

print("First 5 vectors:")
print(data[:5])

print("\nMean vector:")
print(mu)
from gaussian import D1, D2, mu1, mu2

# Combine D1 and D2 to form 2-D data
data = []
for i in range(len(D1)):
    data.append([D1[i], D2[i]])

# Mean vector
mu = [mu1, mu2]

print("First 5 vectors:")
print(data[:5])

print("\nMean vector:")
print(mu)
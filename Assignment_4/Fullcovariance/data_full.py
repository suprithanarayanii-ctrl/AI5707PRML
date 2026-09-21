import numpy as np
from gaussian import D1, D2
from math import sqrt

# Mean
mu1 = 5
mu2 = 10

# Full covariance matrix
# C = [ 4  1 ]
#     [ 1  1 ]
C = np.array([
    [4, 1],
    [1, 1]
], dtype=float)

n = len(D1)

# --------------------------------
# 1. Eigenvalues and Eigenvectors
# --------------------------------
eigenvalues, eigenvectors = np.linalg.eigh(C)

lambda1 = eigenvalues[1]
lambda2 = eigenvalues[0]

v1 = eigenvectors[:, 1]
v2 = eigenvectors[:, 0]

print("Eigenvalues of C:")
print("Lambda 1 =", lambda1)
print("Lambda 2 =", lambda2)

print("\nEigenvectors of C:")
print("v1 =", v1)
print("v2 =", v2)

# --------------------------------
# 2. Calculate C^(1/2)
# --------------------------------
sqrt_lambda1 = sqrt(lambda1)
sqrt_lambda2 = sqrt(lambda2)
C_sqrt = (
    sqrt_lambda1 * np.outer(v1, v1)
    + sqrt_lambda2 * np.outer(v2, v2)
)

print("\nC^(1/2):")
print(C_sqrt)

# --------------------------------
# 3. Generate Y
# Y = mu + C^(1/2)(X - mu)
# --------------------------------
Y1 = []
Y2 = []
for i in range(n):
    X = np.array([
        D1[i] - mu1,
        D2[i] - mu2
    ])
    Y = np.array([mu1, mu2]) + C_sqrt @ X
    Y1.append(Y[0])
    Y2.append(Y[1])

# --------------------------------
# 4. Create 2-D data
# --------------------------------
data = []
for i in range(n):
    data.append([Y1[i], Y2[i]])
mu = [mu1, mu2]
print("\nFirst 5 vectors:")
print(data[:5])
print("\nMean vector:")
print(mu)
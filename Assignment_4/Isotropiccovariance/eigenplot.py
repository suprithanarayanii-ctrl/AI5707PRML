import matplotlib.pyplot as plt
from math import sqrt
from data import mu
from covariance import covariance
import numpy as np

A = np.array(covariance)
eigenvalues, eigenvectors = np.linalg.eig(A)
if eigenvalues[0] < eigenvalues[1]:
    eigenvalues[0], eigenvalues[1] = eigenvalues[1], eigenvalues[0]
    eigenvectors[:, [0, 1]] = eigenvectors[:, [1, 0]]

lambda1 = eigenvalues[0]
lambda2 = eigenvalues[1]

# Eigenvector corresponding to Lambda 1
x1 = eigenvectors[0][0]
y1 = eigenvectors[1][0]

# Eigenvector corresponding to Lambda 2
x2 = eigenvectors[0][1]
y2 = eigenvectors[1][1]

print("\nEigenvector 1:")
print("[", x1, "]")
print("[", y1, "]")

print("\nEigenvector 2:")
print("[", x2, "]")
print("[", y2, "]")
# Length of eigenvectors
length1 = sqrt(x1**2 + y1**2)
length2 = sqrt(x2**2 + y2**2)

# Normalize eigenvectors
x1 = x1 / length1
y1 = y1 / length1
x2 = x2 / length2
y2 = y2 / length2

print("\nNormalized Eigenvector 1:")
print("[", x1, "]")
print("[", y1, "]")
print("\nNormalized Eigenvector 2:")
print("[", x2, "]")
print("[", y2, "]")

# -------------------------------
# Major and minor axis lengths
# -------------------------------
major_axis = sqrt(lambda1)
minor_axis = sqrt(lambda2)

print("\nMajor Axis Length =", major_axis)
print("Minor Axis Length =", minor_axis)

# -------------------------------
# Plot eigenvectors
# -------------------------------
plt.figure()

# Major axis
plt.plot(
    [mu[0] - major_axis * x1,
     mu[0] + major_axis * x1],
    [mu[1] - major_axis * y1,
     mu[1] + major_axis * y1],
    linewidth=2,
    label="Major Axis"
)

# Minor axis
plt.plot(
    [mu[0] - minor_axis * x2,
     mu[0] + minor_axis * x2],
    [mu[1] - minor_axis * y2,
     mu[1] + minor_axis * y2],
    linewidth=2,
    label="Minor Axis"
)

# Mean point
plt.scatter(
    mu[0],
    mu[1],
    marker='x',
    s=100,
    label="Mean"
)

plt.xlabel("x1 (D1)")
plt.ylabel("x2 (D2)")
plt.title("Eigenvectors and Principal Axes")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()
from gaussian import gaussian
import numpy as np
import matplotlib.pyplot as plt

# Number of samples
n = 5000

# Create grid
x = np.linspace(-2, 10, 400)
y = np.linspace(-2, 10, 400)
X, Y = np.meshgrid(x, y)

# ============================================================
# DISCRIMINANT FUNCTION
# ============================================================
def discriminant(x, y, mu, sigma_x, sigma_y):
    term1 = ((x - mu[0]) ** 2) / (sigma_x ** 2)
    term2 = ((y - mu[1]) ** 2) / (sigma_y ** 2)
    return -0.5 * (term1 + term2) \
           - np.log(sigma_x * sigma_y)

# ============================================================
# CASE 1: C1 = C2 = sigma^2 I
# ============================================================
mu1 = [2, 2]
mu2 = [6, 6]

sigma1_x = 1
sigma1_y = 1

sigma2_x = 1
sigma2_y = 1

# Generate Class 1
X1 = gaussian(mu1[0], sigma1_x, n)
Y1 = gaussian(mu1[1], sigma1_y, n)

# Generate Class 2
X2 = gaussian(mu2[0], sigma2_x, n)
Y2 = gaussian(mu2[1], sigma2_y, n)

# Discriminants
G1 = discriminant(X, Y, mu1, sigma1_x, sigma1_y)
G2 = discriminant(X, Y, mu2, sigma2_x, sigma2_y)
D = G1 - G2

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X1, Y1, alpha=0.3, label="Class ω1")
plt.scatter(X2, Y2, alpha=0.3, label="Class ω2")
plt.contour(X, Y, D, levels=[0], linewidths=2)
plt.scatter(mu1[0], mu1[1],
            marker='x', s=100, label="Mean ω1")
plt.scatter(mu2[0], mu2[1],
            marker='x', s=100, label="Mean ω2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Case 1: C1 = C2 = σ²I")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# CASE 2: C1 = C2 = C
# C IS DIAGONAL
# ============================================================
mu1 = [2, 2]
mu2 = [6, 6]

sigma1_x = 2
sigma1_y = 1

sigma2_x = 2
sigma2_y = 1

# Generate Class 1
X1 = gaussian(mu1[0], sigma1_x, n)
Y1 = gaussian(mu1[1], sigma1_y, n)

# Generate Class 2
X2 = gaussian(mu2[0], sigma2_x, n)
Y2 = gaussian(mu2[1], sigma2_y, n)

# Discriminants
G1 = discriminant(X, Y, mu1, sigma1_x, sigma1_y)
G2 = discriminant(X, Y, mu2, sigma2_x, sigma2_y)
D = G1 - G2

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X1, Y1, alpha=0.3, label="Class ω1")
plt.scatter(X2, Y2, alpha=0.3, label="Class ω2")
plt.contour(X, Y, D, levels=[0], linewidths=2)
plt.scatter(mu1[0], mu1[1],
            marker='x', s=100, label="Mean ω1")
plt.scatter(mu2[0], mu2[1],
            marker='x', s=100, label="Mean ω2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Case 2: C1 = C2 = C (Diagonal)")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# CASE 3: C1 != C2
# ============================================================
mu1 = [2, 2]
mu2 = [6, 6]

# Class 1
sigma1_x = 1
sigma1_y = 1

# Class 2
sigma2_x = 2
sigma2_y = 1

# Generate Class 1
X1 = gaussian(mu1[0], sigma1_x, n)
Y1 = gaussian(mu1[1], sigma1_y, n)

# Generate Class 2
X2 = gaussian(mu2[0], sigma2_x, n)
Y2 = gaussian(mu2[1], sigma2_y, n)

# Discriminants
G1 = discriminant(X, Y, mu1, sigma1_x, sigma1_y)
G2 = discriminant(X, Y, mu2, sigma2_x, sigma2_y)
D = G1 - G2

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X1, Y1, alpha=0.3, label="Class ω1")
plt.scatter(X2, Y2, alpha=0.3, label="Class ω2")
plt.contour(X, Y, D, levels=[0], linewidths=2)
plt.scatter(mu1[0], mu1[1],
            marker='x', s=100, label="Mean ω1")
plt.scatter(mu2[0], mu2[1],
            marker='x', s=100, label="Mean ω2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Case 3: C1 != C2")
plt.legend()
plt.grid(True)
plt.show()
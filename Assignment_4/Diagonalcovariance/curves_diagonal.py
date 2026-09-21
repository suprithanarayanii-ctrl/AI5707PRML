import matplotlib.pyplot as plt
from data_diagonal import data, mu
from covariance_diagonal import covariance

a = covariance[0][0]
b = covariance[0][1]
d = covariance[1][1]

# Inverse of covariance matrix
det = a * d - b * b
inv = [
    [d / det, -b / det],
    [-b / det, a / det]
]

# Grid
x_values = []
y_values = []
for i in range(100):
    x_values.append(mu[0] - 6 + 12 * i / 99)
    y_values.append(mu[1] - 4 + 8 * i / 99)

# Calculate (x - mu)^T C^-1 (x - mu)
Z = []
for y in y_values:
    row = []
    for x in x_values:
        dx = x - mu[0]
        dy = y - mu[1]
        value = (
            dx * (inv[0][0] * dx + inv[0][1] * dy)
            +
            dy * (inv[1][0] * dx + inv[1][1] * dy)
        )
        row.append(value)
    Z.append(row)

plt.figure()
plt.contour(
    x_values,
    y_values,
    Z,
    levels=[1, 2, 3, 4, 5]
)

plt.scatter(
    mu[0],
    mu[1],
    marker='x',
    s=100
)

plt.title("Constant Density Curves")
plt.xlabel("x1 (D1)")
plt.ylabel("x2 (D2)")
plt.grid(True)
plt.show()
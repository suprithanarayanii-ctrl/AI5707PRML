from data_diagonal import data, mu
from math import sqrt

# --------------------------------
# 1. Covariance matrix A
# --------------------------------
n = len(data)

sum_x1 = 0
sum_x2 = 0
sum_x1x2 = 0

for i in range(n):
    x1 = data[i][0]
    x2 = data[i][1]
    sum_x1 += (x1 - mu[0]) ** 2
    sum_x2 += (x2 - mu[1]) ** 2
    sum_x1x2 += (x1 - mu[0]) * (x2 - mu[1])

a = sum_x1 / (n - 1)
b = sum_x1x2 / (n - 1)
d = sum_x2 / (n - 1)

covariance = [
    [a, b],
    [b, d]
]

print("Covariance Matrix A:")
print("[", a, b, "]")
print("[", b, d, "]")
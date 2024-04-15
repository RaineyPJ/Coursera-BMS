import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

# We can use a Cholesky decomposition to convert uncorrelated random processes into correlated ones

# Say we want some process noise u which has a covariance matrix [[1, 0.5], [0.5, 1]]

Sigma_u = np.array([[1, 0.5], [0.5, 1]])
print("Sigma_u: ")
print(Sigma_u)

L = np.linalg.cholesky(Sigma_u).T
print("L: ")
print(L)

print("L.T @ L")
print(L.T @ L)


# Let's generate 5000 instances of uncorrelated white noise X = (x_1, x_2)
rng = np.random.default_rng()
N = 30000
X = rng.standard_normal((2, N))
# print(X)

x_bar = X.mean(axis=1)
print("x_bar: ")
print(x_bar)

# covariance is the mean of the square of variation from the mean
# X is zero mean, so we don't need to worry about subtracting the mean
# X.T @ X will sum over the N realisations, so combines with a 1/N we get the mean
Sigma_X = 1 / N * X @ X.T
print("Sigma_X: ")
print(Sigma_X)

Y = L.T @ X

Sigma_Y = 1 / N * Y @ Y.T
print("Sigma_Y: ")
print(Sigma_Y)

# plt.scatter(Y[0,:], Y[1,:], 1)
# plt.show()

# Now let's make Y with none zero mean
Y = np.array([[1, 2]]).T + Y
plt.scatter(Y[0,:], Y[1,:], 1)
plt.show()

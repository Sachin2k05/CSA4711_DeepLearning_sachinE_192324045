import numpy as np

# Medical image feature matrix
X = np.array([[1, 2],
              [3, 4]])

# Weight matrix
W = np.array([[0.2, 0.3],
              [0.4, 0.5]])

# Matrix multiplication
Z = np.dot(X, W)

# Bias addition
b = np.array([0.1, 0.2])
Z = Z + b

# Element-wise multiplication
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[2, 3],
              [4, 5]])

result = A * B

# Sigmoid activation
Y = 1 / (1 + np.exp(-Z))

print("Matrix Multiplication and Bias:", Z)
print("Element-wise Multiplication:", result)
print("Final Output:", Y)

# Classification
if np.mean(Y) >= 0.5:
    print("Prediction: Abnormal")
else:
    print("Prediction: Normal")
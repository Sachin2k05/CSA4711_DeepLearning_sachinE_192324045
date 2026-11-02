import numpy as np

# Grayscale image (2x2)
image = np.array([[1, 2],
                  [3, 4]])

# Flatten the image into a vector
X = image.flatten()

# Weight matrix
W = np.array([[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6],
              [0.7, 0.8]])

# Bias
b = np.array([0.1, 0.2])

# Matrix multiplication and bias addition
Z = np.dot(X, W) + b

# ReLU activation
Y = np.maximum(0, Z)

print("Flattened Image:", X)
print("Output:", Y)
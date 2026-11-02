import numpy as np

# 500 products with 20 features
P = np.random.rand(500, 20)

# Select two products
A = P[0]
B = P[1]

# Dot product
similarity = np.dot(A, B)

print("Dot Product Similarity:", similarity)

# Matrix multiplication for all products
S = np.dot(P, P.T)

print("Similarity Matrix Shape:", S.shape)

# Recommend similar products to product 1
recommendations = np.argsort(S[0])[::-1][1:6]

print("Recommended Product IDs:", recommendations)
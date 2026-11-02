import numpy as np

# Features of 100 students
# Attendance, Assignment, Internal Marks
X = np.random.randint(40, 101, (100, 3))

# Weight matrix
W = np.array([[0.4],
              [0.3],
              [0.3]])

# Bias
b = np.array([5])

# Matrix multiplication
Z = np.dot(X, W) + b

# Prediction
Y = np.maximum(0, Z)

print("Student Feature Matrix Shape:", X.shape)
print("Predicted Performance:")
print(Y)
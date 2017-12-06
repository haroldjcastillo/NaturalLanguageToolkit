import numpy as np

# V1, V2, V3 ... Vn in R^n
# V1 + V2  + ... Vn but we scale by arbitrary constants C1V1 + C2V2 + ... CnVn

x = np.array([[1, 2], [0, 3]])
y = ([3, 2])
scalars = np.linalg.solve(x, y)
print(scalars)

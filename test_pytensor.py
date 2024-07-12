import pytensor
import pytensor.tensor as at

# Define a simple function
x = at.vector('x')
y = x ** 2
f = pytensor.function([x], y)

# Test the function
import numpy as np
print(f(np.array([1, 2, 3, 4])))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate random data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Plotting the data and the model's predictions
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Linear regression line')
plt.xlabel('X (independent variable)')
plt.ylabel('y (dependent variable)')
plt.title('Simple Linear Regression Example')
plt.legend()
plt.show()

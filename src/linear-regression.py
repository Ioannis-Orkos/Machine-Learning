import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate some random data
def generate_data():
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y


# Train a linear regression model on the data
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# Plot the data and the linear regression line
def plot_regression(X, y, model):
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, model.predict(X), color='red', label='Linear regression line')
    plt.xlabel('X (independent variable)')
    plt.ylabel('y (dependent variable)')
    plt.title('Simple Linear Regression Example')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, y = generate_data()
    model = train_linear_regression(X, y)
    plot_regression(X, y, model)
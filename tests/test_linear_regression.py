import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.linear_regression import generate_data, train_linear_regression

class TestLinearRegression(unittest.TestCase):

    def test_generate_data(self):
        X, y = generate_data()
        self.assertEqual(X.shape, (100, 1))  # Check if X has the correct shape
        self.assertEqual(y.shape, (100, 1))  # Check if y has the correct shape

    def test_model_training(self):
        X, y = generate_data()
        model = train_linear_regression(X, y)

        # Check if the model coefficients are within expected ranges
        self.assertAlmostEqual(model.coef_[0][0], 3, delta=1)
        self.assertAlmostEqual(model.intercept_[0], 4, delta=2)

if __name__ == '__main__':
    unittest.main()
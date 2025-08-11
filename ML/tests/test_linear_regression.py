import unittest

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from linear_regression import train, eval

EXPECTED = {
    "coef": 0.4203217769894005,
    "intercept": 0.4432063522765708,
    "rmse": 0.8494105152406937,
    "mse": 0.7214982234014606,
    "r2": 0.4466846804895943,
}

TOL = 1e-2

class TestLinearRegression(unittest.TestCase):
    def test_train_learns_coefficients(self):
        rng = np.random.default_rng(42)
        X = rng.random((300, 1))
        true_coef, true_intercept = 4.2, -1.3
        y = true_coef * X.squeeze() + true_intercept

        model = train(X, y)

        self.assertIsInstance(model, LinearRegression)
        self.assertAlmostEqual(model.coef_[0], true_coef, delta=TOL)
        self.assertAlmostEqual(model.intercept_, true_intercept, delta=TOL)

    def test_eval_returns_perfect_metrics(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 2.0, 4.0, 6.0])

        perfect_model = LinearRegression().fit(X, y)
        metrics = eval(perfect_model, X, y)

        self.assertSetEqual(set(metrics), {"rmse", "mse", "r2"})
        self.assertAlmostEqual(metrics["rmse"], 0.0, delta=TOL)
        self.assertAlmostEqual(metrics["mse"], 0.0, delta=TOL)
        self.assertAlmostEqual(metrics["r2"], 1.0, delta=TOL)

    def test_california_exact_values(self):
        data = fetch_california_housing()
        X = data.data[:, [data.feature_names.index("MedInc")]]
        y = data.target

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=0, shuffle=True
        )

        model = train(X_train, y_train)
        metrics = eval(model, X_val, y_val)

        self.assertAlmostEqual(model.coef_[0], EXPECTED["coef"], delta=TOL,
                               msg="Coefficient drift")
        self.assertAlmostEqual(model.intercept_, EXPECTED["intercept"], delta=TOL,
                               msg="Intercept drift")

        self.assertAlmostEqual(metrics["rmse"], EXPECTED["rmse"], delta=TOL,
                               msg="RMSE drift")
        self.assertAlmostEqual(metrics["mse"], EXPECTED["mse"], delta=TOL,
                               msg="MSE drift")
        self.assertAlmostEqual(metrics["r2"], EXPECTED["r2"], delta=TOL,
                               msg="R² drift")

if __name__ == "__main__":
    unittest.main()

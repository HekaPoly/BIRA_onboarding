import unittest

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from polynomial_regression import (
    train_poly,
    eval_poly,
    train_linear,
    eval_linear,
    generate_polynomial_data,
)

LIN_EXPECTED = {
    "rmse": 5.4827,
    "mse": 30.0603,
    "r2": 0.7305,
}

POLY_EXPECTED = {
    "rmse": 3.0354,
    "mse": 9.2138,
    "r2": 0.9174,
}

TOL = 1e-2

class TestPolynomialRegression(unittest.TestCase):
    def test_train_poly_perfect_fit(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.5, 1.5, size=(300, 1))
        x = X[:, 0]
        y = 3 * x**3 - 2 * x**2 + x

        model, poly = train_poly(X, y, degree=3)
        metrics = eval_poly(model, poly, X, y)

        self.assertIsInstance(model, LinearRegression)
        self.assertIsInstance(poly, PolynomialFeatures)

        self.assertAlmostEqual(metrics["rmse"], 0.0, delta=1e-9)
        self.assertAlmostEqual(metrics["mse"],  0.0, delta=1e-9)
        self.assertAlmostEqual(metrics["r2"],   1.0, delta=1e-9)

    def test_polynomial_beats_linear_on_synthetic_data(self):
        X, y = generate_polynomial_data(n_samples=500, noise=3, random_state=0)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=0, shuffle=True
        )

        lin_model  = train_linear(X_tr, y_tr)
        lin_metrics = eval_linear(lin_model, X_val, y_val)

        poly_model, poly = train_poly(X_tr, y_tr, degree=3)
        poly_metrics = eval_poly(poly_model, poly, X_val, y_val)

        self.assertLess(poly_metrics["rmse"], lin_metrics["rmse"],
                        msg="Le modèle polynomial devrait réduire le RMSE.")
        self.assertGreater(poly_metrics["r2"], lin_metrics["r2"],
                           msg="Le modèle polynomial devrait augmenter le R².")

        for key, expected in LIN_EXPECTED.items():
            self.assertAlmostEqual(lin_metrics[key], expected, delta=TOL,
                                   msg=f"Dérive détectée pour {key} (linéaire).")
        for key, expected in POLY_EXPECTED.items():
            self.assertAlmostEqual(poly_metrics[key], expected, delta=TOL,
                                   msg=f"Dérive détectée pour {key} (polynomial).")

if __name__ == "__main__":
    unittest.main()
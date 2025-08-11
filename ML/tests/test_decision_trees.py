import unittest

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from decision_trees import train, eval, create_model

EXPECTED = {
    "LinearRegression": {"rmse": 0.7273, "mse": 0.5290, "r2": 0.5943},
    "DecisionTree":     {"rmse": 0.7290, "mse": 0.5315, "r2": 0.5924},
    "RandomForest":     {"rmse": 0.5131, "mse": 0.2633, "r2": 0.7981},
    "GradientBoosting": {"rmse": 0.5388, "mse": 0.2903, "r2": 0.7774},
}

TOL = 1e-2

class TestTreeBasedModels(unittest.TestCase):
    def test_linear_regression_perfect_fit(self):
        rng = np.random.default_rng(1)
        X = rng.random((80, 3))
        coeff = rng.random(3)
        y = X @ coeff
        model = train(LinearRegression, X, y)
        metrics = eval(model, X, y)
        for key in ("rmse", "mse"):
            self.assertAlmostEqual(metrics[key], 0.0, delta=TOL)
        self.assertAlmostEqual(metrics["r2"], 1.0, delta=TOL)

    def test_decision_tree_perfect_fit(self):
        rng = np.random.default_rng(2)
        X = rng.random((100, 2))
        y = rng.random(100)
        model = train(DecisionTreeRegressor, X, y, random_state=0)
        metrics = eval(model, X, y)
        for key in ("rmse", "mse"):
            self.assertAlmostEqual(metrics[key], 0.0, delta=TOL)
        self.assertAlmostEqual(metrics["r2"], 1.0, delta=TOL)

    def test_random_forest_perfect_fit(self):
        rng = np.random.default_rng(3)
        X = rng.random((60, 4))
        y = rng.random(60)
        model = train(
            RandomForestRegressor,
            X,
            y,
            n_estimators=1,
            bootstrap=False,
            random_state=0,
        )
        metrics = eval(model, X, y)
        for key in ("rmse", "mse"):
            self.assertAlmostEqual(metrics[key], 0.0, delta=TOL)
        self.assertAlmostEqual(metrics["r2"], 1.0, delta=TOL)

    def test_gradient_boosting_perfect_fit(self):
        rng = np.random.default_rng(4)
        X = rng.random((50, 2))
        y = rng.random(50)
        model = train(
            GradientBoostingRegressor,
            X,
            y,
            n_estimators=500,
            learning_rate=1.0,
            max_depth=3,
            random_state=0,
        )
        metrics = eval(model, X, y)
        for key in ("rmse", "mse"):
            self.assertAlmostEqual(metrics[key], 0.0, delta=TOL)
        self.assertAlmostEqual(metrics["r2"], 1.0, delta=TOL)

    def test_expected_metrics_california(self):
        data = fetch_california_housing()
        X, y = data.data, data.target
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=0,
            shuffle=True,
        )
        suite = [
            ("LinearRegression", LinearRegression, {}),
            ("DecisionTree", DecisionTreeRegressor, {"random_state": 0}),
            (
                "RandomForest",
                RandomForestRegressor,
                {"n_estimators": 100, "random_state": 0},
            ),
            (
                "GradientBoosting",
                GradientBoostingRegressor,
                {"n_estimators": 100, "random_state": 0},
            ),
        ]
        for name, cls, kwargs in suite:
            model = train(cls, X_tr, y_tr, **kwargs)
            metrics = eval(model, X_val, y_val)
            for key in ("rmse", "mse", "r2"):
                self.assertAlmostEqual(metrics[key], EXPECTED[name][key], delta=TOL)

if __name__ == "__main__":
    unittest.main()
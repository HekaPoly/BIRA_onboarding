import unittest

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from logistic_regression import train, eval

EXPECTED = {
    "Accuracy":  0.8684,
    "Precision": 0.9014,
    "Recall":    0.8889,
    "F1":        0.8951,
    "conf_mat":  np.array([[35, 7],
                           [ 8, 64]]),
}

TOL = 1e-2

class TestLogisticRegression(unittest.TestCase):
    def test_train_perfect_separation(self):
        rng = np.random.default_rng(0)
        X_pos = rng.normal(loc=+2.0, scale=0.2, size=(50, 2))
        X_neg = rng.normal(loc=-2.0, scale=0.2, size=(50, 2))
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(50), np.zeros(50)])

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        model = train(X_std, y)
        metrics = eval(model, X_std, y)

        self.assertIsInstance(model, LogisticRegression)
        for m in ("Accuracy", "Precision", "Recall", "F1"):
            self.assertAlmostEqual(metrics[m], 1.0, delta=1e-6,
                                   msg=f"{m} should be 1.0 on separable data")

    def test_breast_cancer_expected_metrics(self):
        data = load_breast_cancer()
        features = ["mean radius", "mean texture"]
        idx = [list(data.feature_names).index(f) for f in features]
        X = data.data[:, idx]
        y = data.target

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )

        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr)
        X_val_std = scaler.transform(X_val)

        model = train(X_tr_std, y_tr)
        metrics = eval(model, X_val_std, y_val)

        for key in ("Accuracy", "Precision", "Recall", "F1"):
            self.assertAlmostEqual(metrics[key], EXPECTED[key], delta=TOL,
                                   msg=f"Dérive détectée pour {key}")

        conf = confusion_matrix(y_val, model.predict(X_val_std))
        self.assertTrue(np.array_equal(conf, EXPECTED["conf_mat"]),
                        msg="Matrice de confusion différente des valeurs de référence")

if __name__ == "__main__":
    unittest.main()

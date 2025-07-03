import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple

def generate_polynomial_data(n_samples: int, noise: int, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère un dataset suivant une courbe polynomiale
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-2, 2, size=(n_samples, 1))
    x = X[:, 0]
    y = 3 * x**3 - 2 * x**2 + x + noise * rng.normal(size=n_samples)
    return X, y

def train_poly(X_train: np.ndarray, y_train: np.ndarray, degree: int = 2) -> Tuple[LinearRegression, PolynomialFeatures]:
    """
    Crée et entraîne un modèle de régression polynomiale.
    Retourne le modèle et le transformateur PolynomialFeatures.
    """
    # TODO: Implémenter l'entraînement du modèle de régression polynomiale

    raise NotImplementedError()

def eval_poly(model: LinearRegression, poly: PolynomialFeatures, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    """
    Évalue le modèle de régression polynomiale sur l'ensemble de validation transformé et renvoie les métriques.
    """
    # TODO: Implémenter la validation du modèle de régression polynomiale

    raise NotImplementedError()

def train_linear(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Entraîne un modèle de régression linéaire sur les données d'origine.
    """
    # TODO: Implémenter l'entraînement du modèle de régression linéaire

    raise NotImplementedError()

def eval_linear(model: LinearRegression, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    """
    Évalue le modèle de régression linéaire sur l'ensemble de validation et renvoie les métriques.
    """
    # TODO: Implémenter la validation du modèle de régression linéaire

    raise NotImplementedError()

def plot_comparison(model_linear: LinearRegression, model_poly: LinearRegression, poly: PolynomialFeatures, 
                    X: np.ndarray, y: np.ndarray, degree: int):
    """
    Trace les courbes de prédiction des modèles linéaire et polynomial sur une plage de valeurs, 
    ainsi que les données brutes pour comparaison.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(X, y, s=10, alpha=0.4, label="Données brutes")
    
    x_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_line_linear = model_linear.predict(x_line)
    x_line_poly = poly.transform(x_line)
    y_line_poly = model_poly.predict(x_line_poly)
    
    plt.plot(x_line, y_line_linear, color="blue", lw=2, label="Régression linéaire")
    plt.plot(x_line, y_line_poly, color="red", lw=2, label=f"Régression polynomiale (deg {degree})")
    plt.xlabel("Variable explicative")
    plt.ylabel("Cible")
    plt.title("Comparaison: régression linéaire vs polynomiale")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(test_ratio: float = 0.2, random_state: int = 0, degree: int = 2):
    X, y = generate_polynomial_data(n_samples=500, noise=3)

    # TODO: Séparer le jeu de données en ensembles d'entraînement et de validation
    
    # TODO: Entraînement et évaluation du modèle de régression linéaire
    
    # TODO: Entraînement et évaluation du modèle de régression polynomiale
    
    # TODO: Afficher les métriques du modèle de régression linéaire
    
    # TODO: Afficher les métriques du modèle de régression polynomiale
    
    # TODO: Afficher les visualisations de comparaison des deux modèles
    #       à l'aide de la fonction plot

    raise NotImplementedError()

if __name__ == "__main__":
    main(degree=3)

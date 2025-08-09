import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score

def train(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Entraîne un modèle de régression linéaire sur les données fournies.

    Parameters
    ----------
    X_train : np.ndarray
        Matrice des variables explicatives d'entraînement, 
        de forme (n_samples, n_features).
    y_train : np.ndarray
        Vecteur des valeurs cibles d'entraînement, 
        de forme (n_samples,).

    Returns
    -------
    model : LinearRegression
        Modèle de régression linéaire ajusté sur les données.
    """
    # TODO: Effectuer la régression linéaire sur l'ensemble d'entraînement
    raise NotImplementedError()

def eval(model: LinearRegression, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    """
    Évalue les performances du modèle sur un ensemble de validation.

    Parameters
    ----------
    model : LinearRegression
        Modèle de régression linéaire préalablement entraîné.
    X_val : np.ndarray
        Matrice des variables explicatives de validation,
        de forme (n_samples, n_features).
    y_val : np.ndarray
        Vecteur des valeurs cibles de validation,
        de forme (n_samples,).

    Returns
    -------
    metrics : dict[str, float]
        Dictionnaire contenant les métriques, où les clés sont :

        - ``rmse`` : erreur quadratique moyenne racine.
        - ``mse``  : erreur quadratique moyenne.
        - ``r2``   : coefficient de détermination R².
    """
    # TODO: Effectuer la validation du modèle sur l'ensemble de validation
    raise NotImplementedError()

def plot(model: LinearRegression, X: np.ndarray, y: np.ndarray):
    plt.figure(figsize=(7, 5))
    plt.scatter(X, y, s=10, alpha=0.4, label="Données brutes")
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color="red", lw=2, label="Droite de régression")
    plt.xlabel("Revenu médian")
    plt.ylabel("Prix médian d'un maison")
    plt.title("Le marché immobilier californien - régression linéaire simple")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(test_ratio: float = 0.2, random_state: int = 0):
    data = fetch_california_housing()
    X = data.data[:, [data.feature_names.index("MedInc")]]
    y = data.target

    # TODO: Séparer le jeu de données en ensembles d'entraînement et de validation

    # TODO: Instancier et entraîner votre modèle et effectuer la validation du modèle

    # TODO: Afficher les métriques, les coefficients et l'ordonnée à l'origine
    #       du modèle

    # TODO: Afficher la visualisation du jeu de données avec la droite de
    #       régression linéaire à l'aide de la fonction plot
    raise NotImplementedError()

if __name__ == "__main__":
    main()
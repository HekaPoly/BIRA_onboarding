import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def train(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Crée et entraîne un modèle de régression logistique.
    Retourne le modèle entraîné.
    """

    # TODO: Effectuer la régression logistique sur l'ensemble d'entraînement

    raise NotImplementedError()

def eval(model: LogisticRegression, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    """
    Évalue le modèle sur l'ensemble de validation et retourne les métriques
    (Accuracy, Precision, Recall, F1).
    """

    # TODO: Effectuer la validation du modèle sur l'ensemble de validation

    raise NotImplementedError()

def plot(model: LogisticRegression, scaler: StandardScaler, X: np.ndarray, y: np.ndarray):
    plt.figure(figsize=(7, 5))
    plt.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap="bwr", alpha=0.6, edgecolor="k", s=35, label="Données"
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid_std = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = model.predict(grid_std).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.15, cmap="bwr")
    plt.xlabel("Rayon moyen")
    plt.ylabel("Texture moyenne")
    plt.title("Cancer du sein - Régression logistique")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(test_ratio: float = 0.2, random_state: int = 42):
    data = load_breast_cancer()
    features = ["mean radius", "mean texture"]
    idx = [list(data.feature_names).index(f) for f in features]
    X = data.data[:, idx]
    y = data.target

    # TODO: Séparer votre jeu de données en ensemble d'entraînement et de validation

    # TODO: Normaliser vos variables explicatives

    # TODO: Instancier votre modèle entraîné et effectuer la validation du modèle

    # TODO: Afficher les métriques et la matrice de confusion

    # TODO: Afficher les visualisations à partir de la fonction plot

    raise NotImplementedError()

if __name__ == "__main__":
    main()

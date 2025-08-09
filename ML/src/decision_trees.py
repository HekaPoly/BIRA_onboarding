import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train(model_class, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
    """
    Crée et entraîne un modèle de régression donné sur les données d'entraînement.

    Parameters
    ----------
    model_class : class
        Classe du modèle à instancier (ex: LinearRegression, DecisionTreeRegressor).
    X_train : np.ndarray
        Matrice des variables explicatives d'entraînement, forme (n_samples, n_features).
    y_train : np.ndarray
        Vecteur des cibles d'entraînement, forme (n_samples,).
    **kwargs :
        Arguments optionnels à passer au constructeur du modèle.

    Returns
    -------
    model : objet
        Modèle entraîné.
    """

    # TODO: Implémenter l'entraînement du modèle
    raise NotImplementedError()

def eval(model, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float | np.ndarray]:
    """
    Évalue un modèle sur un ensemble de validation.

    Calcule les métriques RMSE, MSE et coefficient de détermination R²,
    et génère les prédictions correspondantes.

    Parameters
    ----------
    model : objet
        Modèle entraîné avec une méthode predict.
    X_val : np.ndarray
        Matrice des variables explicatives de validation.
    y_val : np.ndarray
        Vecteur des cibles de validation.

    Returns
    -------
    dict[str, float | np.ndarray]
        Dictionnaire contenant les métriques et les prédictions, où les clés sont:
        - 'rmse' (float) : racine de l'erreur quadratique moyenne,
        - 'mse' (float) : erreur quadratique moyenne,
        - 'r2' (float) : coefficient de détermination,
        - 'y_pred' (np.ndarray) : prédictions sur X_val.
    """
    # TODO: Implémenter la validation du modèle
    raise NotImplementedError()

def plot(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", lw=2, label="y = ŷ")
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_model(name: str, model_class, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, **kwargs):
    print(f"\n----- {name} -----")
    model = train(model_class, X_train, y_train, **kwargs)
    results = eval(model, X_val, y_val)

    print(f"Validation RMSE : {results['rmse']:.4f}")
    print(f"Validation MSE  : {results['mse']:.4f}")
    print(f"Validation R²   : {results['r2']:.4f}")

    plot(y_val, results["y_pred"], f"Prédictions vs Réel - {name}")

def main(test_ratio: float = 0.2, random_state: int = 0):
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # TODO: Séparer votre jeu de données en ensembles d'entraînement et de validation

    # TODO: Instancier, entraîner et valider les modèles suivants à partir de la fonction create_model:
    #       Régression linéaire, Arbre de décision, Forêt aléatoire, Gradient boosting
    #       Indice: Passez la classe du modèle (et non une instance) comme deuxième argument de
    #       create_model.  La fonction se chargera de l’instancier via train(...).
    #       Exemple : create_model("Régression linéaire", LinearRegression, ...).
    #       Ajoutez les hyperparamètres éventuels dans kwargs (après X_val, y_val).
          
    raise NotImplementedError()

if __name__ == "__main__":
    main()

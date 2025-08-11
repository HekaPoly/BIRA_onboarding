import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNet(nn.Module):
    """
    Réseau de neurones entièrement connecté (MLP) pour la classification MNIST.

    Architecture :
    - nn.Flatten() : aplatissement des images 28x28 en vecteurs 1D de taille 784.
    - Couche 1 : 784 → 128 neurones.
    - ReLU : fonction d'activation non linéaire.
    - Couche 2 : 128 → 64 neurones.
    - ReLU : fonction d'activation non linéaire.
    - Couche 3 : 64 → 10 neurones (logits pour chaque classe MNIST).

    Cette architecture permet d'apprendre des représentations successives de plus en plus abstraites,
    adaptées à la classification des chiffres manuscrits.
    """

    def __init__(self):
        super().__init__()
        # TODO: Implémenter l'architecture du réseau de neurones
        self.net = None
        raise NotImplementedError()

    def forward(self, x):
        """
        Propagation avant du réseau.

        Parameters
        ----------
        x : torch.Tensor
            Batch d'images d'entrée, de forme (batch_size, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Logits de sortie, de forme (batch_size, 10).
        """
        # TODO: Implémenter la propagation avant du réseau
        raise NotImplementedError()


def train(model, loader, criterion, optimizer, epoch):
    """
    Entraîne le modèle pour un epoch.

    Parameters
    ----------
    model : nn.Module
        Modèle à entraîner.
    loader : DataLoader
        Itérateur fournissant les batches d'entraînement.
    criterion : torch.nn.Module
        Fonction de perte (CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        Optimiseur (Adam).
    epoch : int
        Numéro de l'epoch en cours (affichage).
    """
    model.train()

    # TODO: Implémenter l'entraînement de votre modèle
    raise NotImplementedError()


def eval(model, loader, criterion):
    """
    Évalue le modèle sur un ensemble de validation.

    Parameters
    ----------
    model : nn.Module
        Modèle entraîné.
    loader : DataLoader
        Itérateur fournissant les batches de validation.
    criterion : torch.nn.Module
        Fonction de perte utilisée (CrossEntropyLoss).

    Returns
    -------
    y_true : np.ndarray
        Labels vrais concaténés sur tout l'ensemble.
    y_pred : np.ndarray
        Prédictions concaténées sur tout l'ensemble.
    """
    model.eval()

    # TODO: Implémenter la validation de votre modèle
    raise NotImplementedError()

def plot_examples(images, labels, preds=None, n=6):
    plt.figure(figsize=(10, 2))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i][0], cmap="gray")
        title = f"{labels[i]}"

        if preds is not None:
            title += f"→{preds[i]}"

        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def main(batch_size=128, epochs=5, lr=1e-3, random_state=0):
    torch.manual_seed(random_state)

    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # TODO: Instancier votre modèle, votre perte (CrossEntropy Loss) et votre optimiseur (Adam)

    # TODO: Effectuer l'entraînement et la validation de votre modèle

    raise NotImplementedError()

    sample_imgs, sample_labels = next(iter(test_loader))[:6]
    sample_preds = model(sample_imgs.to(DEVICE)).argmax(1).cpu()
    plot_examples(sample_imgs, sample_labels, sample_preds)


if __name__ == "__main__":
    main()

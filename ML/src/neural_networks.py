import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNet(nn.Module):
    # TODO: Implémenter l'architecture du réseau de neurones
    def __init__(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


def train(model, loader, criterion, optimizer, epoch):
    model.train()

    # TODO: Implémenter l'entraînement de votre modèle
    raise NotImplementedError()


def eval(model, loader, criterion):
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

    # TODO: Instancier votre modèle, votre perte et votre optimiseur

    # TODO: Effectuer l'entraînement et la validation de votre modèle

    raise NotImplementedError()

    sample_imgs, sample_labels = next(iter(test_loader))[:6]
    sample_preds = model(sample_imgs.to(DEVICE)).argmax(1).cpu()
    plot_examples(sample_imgs, sample_labels, sample_preds)


if __name__ == "__main__":
    main()

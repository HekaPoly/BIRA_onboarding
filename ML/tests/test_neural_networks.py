import unittest

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from neural_networks import NeuralNet, train, eval, DEVICE

TOL = 1e-3

class TestNeuralNetworks(unittest.TestCase):
    def setUp(self):
        self.model = NeuralNet().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        x = torch.randn(10, 1, 28, 28).to(DEVICE)
        y = torch.arange(10).to(DEVICE)
        self.loader = DataLoader(TensorDataset(x, y), batch_size=5)

    def test_forward_output_shape(self):
        for xb, _ in self.loader:
            out = self.model(xb)
            self.assertEqual(out.shape, (xb.size(0), 10))

    def test_train_reduces_loss(self):
        xb, yb = next(iter(self.loader))
        self.model.train()
        initial_loss = self.criterion(self.model(xb), yb).item()

        train(self.model, self.loader, self.criterion, self.optimizer, epoch=1)

        new_loss = self.criterion(self.model(xb), yb).item()
        self.assertLessEqual(new_loss, initial_loss + TOL)

    def test_eval_returns_correct_format(self):
        y_true, y_pred = eval(self.model, self.loader, self.criterion)
        self.assertIsInstance(y_true, np.ndarray)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(len(y_true), len(y_pred))
        self.assertTrue(((y_pred >= 0) & (y_pred < 10)).all())

if __name__ == "__main__":
    unittest.main()
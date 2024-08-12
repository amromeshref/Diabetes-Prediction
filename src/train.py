import os
import sys

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO_DIR_PATH)

from sklearn.metrics import accuracy_score
from src.ANN_model import ANN_MODEL
import numpy as np
from torch.nn import BCELoss
from torch.optim import Adam
import torch
from datetime import datetime

EPOCHS = 500

class Trainer:
    def __init__(self):
        self.criterion = BCELoss()

    def load_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load the data from the data/external directory
        Args:
            None
        Returns:
            X_train: torch.Tensor
                The training data
            y_train: torch.Tensor
                The training labels
            X_test: torch.Tensor
                The testing data
            y_test: torch.Tensor
                The testing labels
        """
        X_train = np.load(os.path.join(REPO_DIR_PATH, "data/processed/X_train.npy"))
        X_train = torch.FloatTensor(X_train)
        y_train = np.load(os.path.join(REPO_DIR_PATH, "data/processed/y_train.npy"))
        y_train = torch.FloatTensor(y_train)
        y_train = y_train.reshape(-1,1)
        X_test = np.load(os.path.join(REPO_DIR_PATH, "data/processed/X_test.npy"))
        X_test = torch.FloatTensor(X_test)
        y_test = np.load(os.path.join(REPO_DIR_PATH, "data/processed/y_test.npy"))
        y_test = torch.FloatTensor(y_test)
        y_test = y_test.reshape(-1,1)
        return X_train, y_train, X_test, y_test
    
    def instantiate_model(self) -> ANN_MODEL:
        """
        Instantiate the model
        Args:
            None
        Returns:
            model: torch.nn.Module
                The model object
        """
        model = ANN_MODEL()
        return model
    
    def instantiate_optimizer(self, model: ANN_MODEL, learning_rate : float = 0.01) -> torch.optim:
        """
        Instantiate the optimizer
        Args:
            model: torch.nn.Module
                The model object
            learning_rate: float
                The learning rate
        Returns:
            optimizer: torch.optim
                The optimizer
        """
        optimizer = Adam(model.parameters(), lr=learning_rate)
        return optimizer

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, model: ANN_MODEL, optimizer, epochs: int = 100) -> list[float]:
        """
        Train the model
        Args:
            X_train: torch.Tensor
                The training data
            y_train: torch.Tensor
                The training labels
            model: torch.nn.Module
                The model object
            optimizer: torch.optim
                The optimizer
            epochs: int
                The number of epochs
        Returns:
            losses: list[float]
                The losses for each epoch
        """
        losses = []
        for epoch in range(1,epochs+1):
            y_pred = model.forward(X_train)
            loss = self.criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} Loss: {loss.item()}")
            losses.append(loss.item())
        return losses

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor, model: ANN_MODEL) -> float:
        """
        Evaluate the model
        Args:
            X_test: torch.Tensor
                The testing data
            y_test: torch.Tensor
                The testing labels
            model: torch.nn.Module
                The model object
        Returns:
            accuracy: float
                The accuracy of the model
        """
        y_pred = model.forward(X_test)
        accuracy = accuracy_score(y_test.detach().numpy(), y_pred.detach().numpy().round())
        return accuracy

if __name__ == "__main__":
    trainer = Trainer()
    X_train, y_train, X_test, y_test = trainer.load_data()
    model = trainer.instantiate_model()
    optimizer = trainer.instantiate_optimizer(model)
    losses = trainer.train(X_train, y_train, model, optimizer, epochs=EPOCHS)
    accuracy_test = trainer.evaluate(X_test, y_test, model)
    accuracy_train = trainer.evaluate(X_train, y_train, model)
    print(f"Accuracy on Training Data: {accuracy_train}")
    print(f"Accuracy on Testing Data: {accuracy_test}")
    torch.save(model, os.path.join(REPO_DIR_PATH, "models/trained_model.pt"))
    with open(os.path.join(REPO_DIR_PATH, "models/logs.txt"), "a") as f:
        f.write(f"Training Accuracy: {accuracy_train} Testing Accuracy: {accuracy_test} Time: {datetime.now()}\n")
    print("Model saved successfully at models/trained_model.pt")

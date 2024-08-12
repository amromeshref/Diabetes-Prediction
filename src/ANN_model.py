import os
import sys

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO_DIR_PATH)


import torch
import torch.nn as nn


class ANN_MODEL(nn.Module):
    def __init__(self, input_dim: int = 8, output_dim: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        y = nn.Sigmoid()(y)
        return y

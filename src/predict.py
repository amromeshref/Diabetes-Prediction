import os
import sys

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO_DIR_PATH)

import torch
import os
import numpy as np

MODEL_PATH = os.path.join(REPO_DIR_PATH, "models/model.pt")

class Predictor:
    def __init__(self):
        self.model = torch.load(MODEL_PATH)
        
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions using the model
        Args:
            x: torch.Tensor
                The input data
        Returns:
            predictions(0/1): np.ndarray
        """
        with torch.no_grad():
            predictions = self.model(x)
        predictions = torch.round(predictions).detach().numpy()
        return predictions
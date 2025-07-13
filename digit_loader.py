from torch import nn
import torch.nn.functional as F

class DigitClassification(nn.Module):
    # Input (784) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(10) → Output

    def __init__(self) -> None:
        super().__init__() 
        self.model = nn.Sequential(
            nn.Linear(784, 128),   # Input layer → Hidden layer 1, 784 inputs and 128 outputs.
            nn.ReLU(),             # Activation function
            nn.Linear(128, 64),    # Hidden layer 1 → Hidden layer 2
            nn.ReLU(),             # Activation function
            nn.Linear(64, 10)      # Hidden layer 2 → Output layer (10 classes)
            
        )
        
    def forward(self, x):
        return self.model(x)
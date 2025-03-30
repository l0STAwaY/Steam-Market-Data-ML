import torch
import random

class LogisticRegressionModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(n_inputs, n_outputs)  # Single output (binary classification)
        self.sigmoid = nn.Sigmoid()  # Sigmoid to map the output between 0 and 1
    
    def forward(self, x):
        x = self.fc(x)  # Linear layer
        x = self.sigmoid(x)  # Apply sigmoid for binary classification
        return x
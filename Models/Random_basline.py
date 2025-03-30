import torch
import random

class RandomClassifier:
    def __init__(self):
        pass
    
    def predict(self, num_samples):
        # Randomly generate 0 or 1 for binary classification
        return torch.tensor([random.randint(0, 1) for _ in range(num_samples)])
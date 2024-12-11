# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# Define distance correlation function (as above)
def distance_correlation(X, Y):
    X_dist = torch.cdist(X.unsqueeze(1), X.unsqueeze(1), p=2)
    Y_dist = torch.cdist(Y.unsqueeze(1), Y.unsqueeze(1), p=2)

    X_mean = X_dist.mean(dim=0, keepdim=True)
    Y_mean = Y_dist.mean(dim=0, keepdim=True)
    X_centered = X_dist - X_mean - X_mean.T + X_dist.mean()
    Y_centered = Y_dist - Y_mean - Y_mean.T + Y_dist.mean()

    dCovXY = (X_centered * Y_centered).mean()
    dCovXX = (X_centered * X_centered).mean()
    dCovYY = (Y_centered * Y_centered).mean()

    dCorr = torch.sqrt(dCovXY / (torch.sqrt(dCovXX * dCovYY) + 1e-10))
    return dCorr

# Neural Network Classifier
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
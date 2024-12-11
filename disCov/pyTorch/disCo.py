# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# %%
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
# %%
# Synthetic data
np.random.seed(42)
X = np.random.rand(1000, 5).astype(np.float32)
Y = (X[:, 0] > 0.5).astype(np.float32)
dijet_mass = X[:, 1]

X_tensor = torch.tensor(X[:, [0, 2, 3, 4]])
Y_tensor = torch.tensor(Y).unsqueeze(1)
dijet_mass_tensor = torch.tensor(dijet_mass)

dataset = TensorDataset(X_tensor, Y_tensor, dijet_mass_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# %%
# Model, loss, optimizer
model = Classifier(input_dim=4)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
lambda_reg = 0.1
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        X_batch, Y_batch, dijet_batch = batch
        optimizer.zero_grad()

        predictions = model(X_batch).squeeze().unsqueeze(1)
        classifier_loss = criterion(predictions, Y_batch)

        # Distance correlation
        dCorr = distance_correlation(predictions, dijet_batch)

        # Combined loss
        loss = classifier_loss + lambda_reg * dCorr
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), "decorrelated_classifier.pth")

# %%
with torch.no_grad():  # Disable gradient calculation for inference
    model.eval()  # Set model to evaluation mode
    predictions = model(X_test)  # Forward pass to get output
    print(predictions)  # These are the predicted continuous values

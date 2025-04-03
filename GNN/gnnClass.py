from torch_geometric.nn import GraphConv, global_mean_pool
import torch.nn as nn
import torch
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN, self).__init__()
        
        # Define the layers
        self.conv1 = GraphConv(input_dim, hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim) 

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        #self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply graph convolutions with BatchNorm
        x = self.conv1(x, edge_index)
        #x = self.bn1(x)  # Batch Normalization after conv1
        x = torch.relu(x)
        
        x = self.conv2(x, edge_index)
        #x = self.bn2(x)  # Batch Normalization after conv2
        x = torch.relu(x)
        
        # Global mean pooling
        x = global_mean_pool(x, data.batch)  # Aggregate node-level features
        
        # Feedforward layers with BatchNorm
        x = self.fc1(x)
        #x = self.bn3(x)  # Batch Normalization after fc1
        x = torch.relu(x)
        
        x = self.fc2(x)  # Final output layer
        
        return x
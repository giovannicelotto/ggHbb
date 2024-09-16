# %%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
# %%
data = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training/Data_1.parquet")
signal = pd.read_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/GluGluHToBB_110.parquet")
data[['dijet_pt', 'dijet_dR']] = data[['dijet_pt', 'dijet_dR']].fillna(0)
signal[['dijet_pt', 'dijet_dR']] = signal[['dijet_pt', 'dijet_dR']].fillna(0)

# Convert the edge feature columns to float explicitly
data[['dijet_pt', 'dijet_dR']] = data[['dijet_pt', 'dijet_dR']].astype(float)
signal[['dijet_pt', 'dijet_dR']] = signal[['dijet_pt', 'dijet_dR']].astype(float)

print(data[['dijet_pt', 'dijet_dR']].dtypes)
print(signal[['dijet_pt', 'dijet_dR']].dtypes)
# %%
jet1_features = ['jet1_pt', 'jet1_eta', 'jet1_btagDeepFlavB']
jet2_features = ['jet2_pt', 'jet2_eta', 'jet2_btagDeepFlavB']
edge_features = ['dijet_pt', 'dijet_dR']

# Load the Cora dataset (Planetoid dataset)
# %%
def create_event_graph(event):
    # Extract node features for jet1 and jet2
    node1_features = event[jet1_features].values
    node2_features = event[jet2_features].values
    
    node_features_array = np.array([node1_features, node2_features], dtype=np.float32)
    node_features = torch.tensor(node_features_array, dtype=torch.float)
    
    # Extract and convert edge features (ensure they're floats)
    edge_features_array = np.array(event[edge_features].astype(float).values, dtype=np.float32)
    edge_features_tensor = torch.tensor(edge_features_array, dtype=torch.float).unsqueeze(0)

    # Edge index: We create a single edge between node 0 (jet1) and node 1 (jet2)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Connect jet1 to jet2

    # Create the graph
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_tensor)

# Apply this function to both datasets
data_graphs = [create_event_graph(row) for idx, row in data.iterrows()]
signal_graphs = [create_event_graph(row) for idx, row in signal.iterrows()]

# %%
# Define the GNN model (2-layer GCN for simplicity)
# Assign labels to graphs
for graph in data_graphs:
    graph.y = torch.tensor([0], dtype=torch.long)  # Label for data

for graph in signal_graphs:
    graph.y = torch.tensor([1], dtype=torch.long)  # Label for signal
# %%
    
# Combine the data and signal graphs
all_graphs = data_graphs + signal_graphs

# Create a PyTorch Geometric DataLoader
train_loader = DataLoader(all_graphs, batch_size=32, shuffle=True)
# %%
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        # Node features (x) and edge indices (edge_index)
        x, edge_index = data.x, data.edge_index
        
        # First convolutional layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Apply dropout for regularization
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second convolutional layer
        x = self.conv2(x, edge_index)
        
        # Use global mean pooling (averaging node embeddings for graph-level output)
        x = torch.mean(x, dim=0)
        
        return F.log_softmax(x, dim=-1)

# Create the model
model = GNN(in_channels=5, hidden_channels=16, out_channels=2)  # 5 input features (jet1 and jet2 features), 2 output classes
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)




# %%

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test():
    model.eval()
    correct = 0
    for data in train_loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(train_loader.dataset)

# Train the model for a few epochs
for epoch in range(100):
    loss = train()
    accuracy = test()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')

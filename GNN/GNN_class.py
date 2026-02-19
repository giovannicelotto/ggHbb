from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv
class JetGraphDataset(InMemoryDataset):
    def __init__(self, graphs_list=None, root=None, transform=None, pre_transform=None):
        """
        graphs_list: list of Data objects (from build_graph)
        root: folder to store dataset (optional)
        """
        super().__init__(root, transform, pre_transform)

        if graphs_list is not None:
            # Collate list of Data objects into a single Data + slices
            self.data, self.slices = self.collate(graphs_list)

    @property
    def raw_file_names(self):
        # Required by PyG, can be empty if preloaded
        return []

    @property
    def processed_file_names(self):
        # Required by PyG, can be empty if preloaded
        return []

    def download(self):
        # Not used
        pass

    def process(self):
        # Not used if graphs_list is given
        pass

class GNN(nn.Module):
    def __init__(self):
        super().__init__()

        edge_mlp = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 5),   # ← dimensione node features
        )

        self.conv1 = GINEConv(edge_mlp, edge_dim=3)
        #self.conv2 = GINEConv(edge_mlp, edge_dim=1)

        self.fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        #x = self.conv2(x, data.edge_index, data.edge_attr)
        x = global_mean_pool(x, data.batch) #[numnodes (3)* BATCH, numfeatures(6)]->[Batch, 6]
        return self.fc(x)



def initialize_mlp(edge_mlp):
    for m in edge_mlp:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # good for ReLU
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return edge_mlp

def initialize_conv(conv1):
    for name, param in conv1.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)  # works well for general conv
            elif "bias" in name:
                nn.init.zeros_(param)
    return conv1
def initialize_fc(fc):
    for m in fc:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # Initialize bias for the last layer to match class balance
            if m.bias is not None:
                if m.out_features == 1:
                    # if positive fraction is p, logit(p) = log(p/(1-p))
                    p = 0.5  # change if classes are unbalanced
                    m.bias.data.fill_(np.log(p/(1-p)))
                else:
                    nn.init.zeros_(m.bias)
    return fc

class GNN_3j1m(nn.Module):
    def __init__(self):
        super().__init__()

        edge_mlp1 = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 32),   # ← dimensione node features
        )

        edge_mlp1 = initialize_mlp(edge_mlp1)

        edge_mlp2 = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32),   # ← dimensione node features
        )

        edge_mlp2 = initialize_mlp(edge_mlp2)



        self.conv1 = GINEConv(edge_mlp1, edge_dim=3)
        self.conv1 = initialize_conv(self.conv1)

        self.conv2 = GINEConv(edge_mlp2, edge_dim=3)
        self.conv2 = initialize_conv(self.conv2)



        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.fc = initialize_fc(self.fc)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = F.relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = F.relu(F.layer_norm(x, x.shape[1:]))
        x_mean = global_mean_pool(x, data.batch)
        x_max  = global_max_pool(x, data.batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = F.layer_norm(x, x.shape[1:]) 
        return self.fc(x)
    

class GNN_3j1m_hetero(nn.Module):
    def __init__(self):
        super().__init__()
        self.jet_encoder = nn.Sequential(
            nn.Linear(5, int(32/2)),
            nn.ReLU(),
            nn.Linear(int(32/2), int(32/2)),
        )
        self.muon_encoder = nn.Sequential(
            nn.Linear(5, int(32/2)), #pt eta phi charge isolation
            nn.ReLU(),
            nn.Linear(int(32/2), int(32/2)),
        )


        edge_mlp1 = nn.Sequential(
            nn.Linear(int(32/2), int(128/2)),
            nn.ReLU(),
            nn.Linear(int(128/2), int(32/2)),   # ← dimensione node features
        )

        edge_mlp1 = initialize_mlp(edge_mlp1)

        edge_mlp2 = nn.Sequential(
            nn.Linear(int(32/2), int(128/2)),
            nn.ReLU(),
            nn.Linear(int(128/2), int(32/2)),   # ← dimensione node features
        )

        edge_mlp2 = initialize_mlp(edge_mlp2)


        self.conv1 = GINEConv(edge_mlp1, edge_dim=4)
        self.conv1 = initialize_conv(self.conv1)

        self.conv2 = GINEConv(edge_mlp2, edge_dim=4)
        self.conv2 = initialize_conv(self.conv2)



        self.fc = nn.Sequential(
            nn.Linear(int(64/2), int(128/2)),
            nn.ReLU(),
            nn.Linear(int(128/2), 1),
        )
        self.fc = initialize_fc(self.fc)

    def forward(self, data):
        device = data.x.device

        jet_mask  = data.type_id == 0
        muon_mask = data.type_id == 1
        x = torch.zeros(data.num_nodes, int(32/2), device=device)
        
        x[jet_mask]  = self.jet_encoder(data.x[jet_mask])
        x[muon_mask] = self.muon_encoder(data.x[muon_mask])

        x = self.conv1(x, data.edge_index, data.edge_attr)
        x = F.relu(F.layer_norm(x, x.shape[1:]))
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = F.relu(F.layer_norm(x, x.shape[1:]))
        x_mean = global_mean_pool(x, data.batch)
        x_max  = global_max_pool(x, data.batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = F.layer_norm(x, x.shape[1:]) 
        return self.fc(x)

class GNN_3j1m_hetero_hetero(nn.Module):
    def __init__(self, coderDimension):
        super().__init__()


        self.jet_encoder = nn.Sequential(
            nn.Linear(5, int(coderDimension)),
            nn.ReLU(),
            #nn.Linear(int(32), int(32)),
            #nn.ReLU(),
        )
        self.muon_encoder = nn.Sequential(
            nn.Linear(5, int(coderDimension)), #pt eta phi charge isolation
            nn.ReLU(),
            #nn.Linear(int(32), int(32)),
            #nn.ReLU(),
        )



        edge_mlp_jetjet1= nn.Sequential(
            nn.Linear(int(coderDimension), int(coderDimension)),
            nn.ReLU(),
            #nn.Linear(int(128), int(32)),   # ← dimensione node features
        )
        edge_mlp_muonjet1= nn.Sequential(
            nn.Linear(int(coderDimension), int(coderDimension)),
            #nn.ReLU(),
            #nn.Linear(int(128), int(32)),   # ← dimensione node features
        )
        self.conv_jetjet1 = GINEConv(edge_mlp_jetjet1, edge_dim=5)
        self.conv_muonjet1 = GINEConv(edge_mlp_muonjet1, edge_dim=4)

        edge_mlp_jetjet2= nn.Sequential(
            nn.Linear(int(coderDimension), int(coderDimension)),
            nn.ReLU(),
            #nn.Linear(int(128), int(32)),   # ← dimensione node features
        )
        edge_mlp_muonjet2= nn.Sequential(
            nn.Linear(int(coderDimension), int(coderDimension)),
            nn.ReLU(),
            #nn.Linear(int(128), int(32)),   # ← dimensione node features
        )
        self.conv_jetjet2 = GINEConv(edge_mlp_jetjet2, edge_dim=5)
        self.conv_muonjet2 = GINEConv(edge_mlp_muonjet2, edge_dim=4)

        #self.conv1 = GINEConv(edge_mlp1, edge_dim=3)
        #self.conv1 = initialize_conv(self.conv1)

        #self.conv2 = GINEConv(edge_mlp2, edge_dim=3)
        #self.conv2 = initialize_conv(self.conv2)



        self.fc = nn.Sequential(
            nn.Linear(int(64), int(64)),
            nn.ReLU(),
            nn.Linear(int(64), 32),
            nn.ReLU(),
            nn.Linear(int(32), 1),
        )
        self.fc = initialize_fc(self.fc)

    def forward(self, data):

        #device = data.x.device
        #print(data.x.shape) # (4*B, num_Features)
        #print(data.type_id.shape)
        jet_mask  = data.type_id == 0 #  4*B
        muon_mask = data.type_id == 1 #  4*B
        #print(data.type_id[:4])
        # data.num_nodes = 4*B
        # preallocate memory for all nodes.
        # The dimension is the dimension after the encoder
        x = torch.zeros(data.num_nodes, int(coderDimension), dtype=torch.float)

        x[jet_mask]  = self.jet_encoder(data.x[jet_mask]) # maps from (4b,numFeatures) to (4b,coderDimension)
        x[muon_mask] = self.muon_encoder(data.x[muon_mask])


        x = self.conv_jetjet1(x, data.edge_index_jetjet, data.edge_attr_jetjet)
        x = self.conv_muonjet1(x, data.edge_index_muonjet, data.edge_attr_muonjet)
        x = F.relu(F.layer_norm(x, x.shape[1:]))
        # Shape is not affected only message passing. Still (4*B, coderDimension)


        x = self.conv_jetjet2(x, data.edge_index_jetjet, data.edge_attr_jetjet)
        x = self.conv_muonjet2(x, data.edge_index_muonjet, data.edge_attr_muonjet)
        x = F.relu(F.layer_norm(x, x.shape[1:]))
        # Shape is not affected only message passing. Still (4*B, coderDimension)


        jet_x  = x[jet_mask]
        muon_x = x[muon_mask]

        jet_mean = global_mean_pool(jet_x, data.batch[jet_mask])
        muon_mean = global_mean_pool(muon_x, data.batch[muon_mask])

        jet_max = global_max_pool(jet_x, data.batch[jet_mask])
        muon_max = global_max_pool(muon_x, data.batch[muon_mask])

        x = torch.cat([muon_mean, jet_mean, jet_max, muon_max, data.u], dim=1)

        

        #x = torch.cat([x_mean, x_max, data.u], dim=1)
        x = F.layer_norm(x, x.shape[1:]) 
        return self.fc(x)
    


class myGAT(nn.Module):
    def __init__(self, coderDimension=8):
        super().__init__()
        self.coder_dim = coderDimension

        self.jet_encoder = nn.Sequential(
            nn.Linear(5, int(coderDimension)),
            nn.ReLU(),
            #nn.Linear(int(32), int(32)),
            #nn.ReLU(),
        )
        self.muon_encoder = nn.Sequential(
            nn.Linear(5, int(coderDimension)), #pt eta phi charge isolation
            nn.ReLU(),
            #nn.Linear(int(32), int(32)),
            #nn.ReLU(),
        )




        self.conv_jetjet1 = GATv2Conv(in_channels=int(coderDimension),
                                        out_channels=int(coderDimension),
                                        heads=1,
                                        concat=False,
                                        edge_dim=5)
        self.conv_muonjet1 = GATv2Conv(
                                in_channels=int(coderDimension),
                                out_channels=int(coderDimension),
                                heads=1,
                                concat=False,
                                edge_dim=4
                            )
        self.conv_jetjet2 = GATv2Conv(
                                in_channels=int(coderDimension),
                                out_channels=int(coderDimension),
                                heads=1,
                                concat=False,
                                edge_dim=5
                            )

        self.conv_muonjet2 = GATv2Conv(
                                in_channels=int(coderDimension),
                                out_channels=int(coderDimension),
                                heads=1,
                                concat=False,
                                edge_dim=4
                            )




        self.fc = nn.Sequential(
            nn.Linear(int(4*self.coder_dim + 32), int(64)),
            nn.ReLU(),
            nn.Linear(int(64), 32),
            nn.ReLU(),
            nn.Linear(int(32), 1),
        )
        self.fc = initialize_fc(self.fc)

    def forward(self, data):

        #device = data.x.device
        #print(data.x.shape) # (4*B, num_Features)
        #print(data.type_id.shape)
        jet_mask  = data.type_id == 0 #  4*B
        muon_mask = data.type_id == 1 #  4*B
        #print(data.type_id[:4])
        # data.num_nodes = 4*B
        # preallocate memory for all nodes.
        # The dimension is the dimension after the encoder
        x = torch.zeros(data.num_nodes, int(self.coder_dim), dtype=torch.float)

        x[jet_mask]  = self.jet_encoder(data.x[jet_mask]) # maps from (4b,numFeatures) to (4b,coderDimension)
        x[muon_mask] = self.muon_encoder(data.x[muon_mask])


        x = (
            self.conv_jetjet1(x, data.edge_index_jetjet, data.edge_attr_jetjet)
          + self.conv_muonjet1(x, data.edge_index_muonjet, data.edge_attr_muonjet)
        )
        x = F.relu(F.layer_norm(x, x.shape[1:]))
        # Shape is not affected only message passing. Still (4*B, coderDimension)


        x = (
            self.conv_jetjet2(x, data.edge_index_jetjet, data.edge_attr_jetjet)
          + self.conv_muonjet2(x, data.edge_index_muonjet, data.edge_attr_muonjet)
        )
        x = F.relu(F.layer_norm(x, x.shape[1:]))
        
        # Shape is not affected only message passing. Still (4*B, coderDimension)
        jet_x  = x[jet_mask]
        muon_x = x[muon_mask]

        jet_mean = global_mean_pool(jet_x, data.batch[jet_mask])
        muon_mean = global_mean_pool(muon_x, data.batch[muon_mask])

        jet_max = global_max_pool(jet_x, data.batch[jet_mask])
        muon_max = global_max_pool(muon_x, data.batch[muon_mask])

        x = torch.cat([muon_mean, jet_mean, jet_max, muon_max, data.u], dim=1)

        

        #x = torch.cat([x_mean, x_max, data.u], dim=1)
        x = F.layer_norm(x, x.shape[1:]) 
        # x has shape 4B, coderDim*4+ 32 features
        return self.fc(x)
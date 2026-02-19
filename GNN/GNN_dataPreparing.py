# %%
import numpy as np
import pandas as pd
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.scaleUnscale import scale_gnn, unscale_gnn
import matplotlib.pyplot as plt
# %%
Xtrain = pd.read_parquet("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/Xtrain.parquet")
Ytrain = np.load("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/Ytrain.npy")
genMassTrain = np.load("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/genMassTrain.npy")
rWtrain = np.load("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/rWtrain.npy")
rWval = np.load("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/rWval.npy")
genMassVal = np.load("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/genMassVal.npy")
Xval = pd.read_parquet("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/Xval.parquet")
Yval = np.load("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/Yval.npy")


Xtrain['jet1_muon_pt_prime'] = Xtrain.jet1_muon_pt/Xtrain.dijet_mass
Xval['jet1_muon_pt_prime'] = Xval.jet1_muon_pt/Xval.dijet_mass

# %%
def mjj(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1  = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2  = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    m2 = (e1 + e2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
    return np.sqrt(np.maximum(m2, 0.))

def ptjj(pt1, phi1, pt2, phi2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)

    ptjj = np.sqrt((px1+px2)**2 + (py1+py2)**2)
    return np.maximum(ptjj, 0.)
# %%
def delta_phi(phi1, phi2):
    """Compute Δφ in [−π, π]"""
    dphi = phi1 - phi2
    dphi = dphi - 2*np.pi*(dphi >= np.pi) + 2*np.pi*(dphi < -np.pi)
    return dphi

Xtrain['mj1j2'] = mjj(Xtrain.jet1_pt, Xtrain.jet1_eta, Xtrain.jet1_phi, Xtrain.jet1_mass,
                      Xtrain.jet2_pt, Xtrain.jet2_eta, Xtrain.jet2_phi, Xtrain.jet2_mass,)
Xtrain['mj1j3'] = mjj(Xtrain.jet1_pt, Xtrain.jet1_eta, Xtrain.jet1_phi, Xtrain.jet1_mass,
                      Xtrain.jet3_pt, Xtrain.jet3_eta, Xtrain.jet3_phi, Xtrain.jet3_mass,)
Xtrain['mj2j3'] = mjj(Xtrain.jet2_pt, Xtrain.jet2_eta, Xtrain.jet2_phi, Xtrain.jet2_mass,
                      Xtrain.jet3_pt, Xtrain.jet3_eta, Xtrain.jet3_phi, Xtrain.jet3_mass,)

Xtrain['ptj1j2'] = ptjj(Xtrain.jet1_pt,  Xtrain.jet1_phi,
                      Xtrain.jet2_pt, Xtrain.jet2_phi)
Xtrain['ptj1j3'] = ptjj(Xtrain.jet1_pt,  Xtrain.jet1_phi,
                      Xtrain.jet3_pt,  Xtrain.jet3_phi,)
Xtrain['ptj2j3'] = ptjj(Xtrain.jet2_pt,  Xtrain.jet2_phi,
                      Xtrain.jet3_pt, Xtrain.jet3_phi,)

Xval['mj1j2'] = mjj(Xval.jet1_pt, Xval.jet1_eta, Xval.jet1_phi, Xval.jet1_mass,
                  Xval.jet2_pt, Xval.jet2_eta, Xval.jet2_phi, Xval.jet2_mass,)
Xval['mj1j3'] = mjj(Xval.jet1_pt, Xval.jet1_eta, Xval.jet1_phi, Xval.jet1_mass,
                      Xval.jet3_pt, Xval.jet3_eta, Xval.jet3_phi, Xval.jet3_mass,)
Xval['mj2j3'] = mjj(Xval.jet2_pt, Xval.jet2_eta, Xval.jet2_phi, Xval.jet2_mass,
                      Xval.jet3_pt, Xval.jet3_eta, Xval.jet3_phi, Xval.jet3_mass,)

Xval['ptj1j2'] = ptjj(Xval.jet1_pt, Xval.jet1_phi,
                      Xval.jet2_pt, Xval.jet2_phi)
Xval['ptj1j3'] = ptjj(Xval.jet1_pt, Xval.jet1_phi,
                      Xval.jet3_pt, Xval.jet3_phi)
Xval['ptj2j3'] = ptjj(Xval.jet2_pt, Xval.jet2_phi,
                      Xval.jet3_pt, Xval.jet3_phi)

Xtrain['dPhi_j1j2'] = np.abs(delta_phi(Xtrain.jet1_phi.values, Xtrain.jet2_phi.values))
Xtrain['dPhi_j1j3'] = np.abs(delta_phi(Xtrain.jet1_phi.values, Xtrain.jet3_phi.values))
Xtrain['dPhi_j2j3'] = np.abs(delta_phi(Xtrain.jet2_phi.values, Xtrain.jet3_phi.values))

Xval['dPhi_j1j2'] = np.abs(delta_phi(Xval.jet1_phi.values, Xval.jet2_phi.values))
Xval['dPhi_j1j3'] = np.abs(delta_phi(Xval.jet1_phi.values, Xval.jet3_phi.values))
Xval['dPhi_j2j3'] = np.abs(delta_phi(Xval.jet2_phi.values, Xval.jet3_phi.values))

Xtrain['dR_j1j2'] = np.sqrt(delta_phi(Xtrain.jet1_phi.values, Xtrain.jet2_phi.values)**2+ (Xtrain.jet1_eta.values - Xtrain.jet2_eta.values)**2)
Xtrain['dR_j1j3'] = np.sqrt(delta_phi(Xtrain.jet1_phi.values, Xtrain.jet3_phi.values)**2+ (Xtrain.jet1_eta.values - Xtrain.jet3_eta.values)**2)
Xtrain['dR_j2j3'] = np.sqrt(delta_phi(Xtrain.jet2_phi.values, Xtrain.jet3_phi.values)**2+ (Xtrain.jet2_eta.values - Xtrain.jet3_eta.values)**2)

Xval['dR_j1j2'] = np.sqrt(delta_phi(Xval.jet1_phi.values, Xval.jet2_phi.values)**2+ (Xval.jet1_eta.values - Xval.jet2_eta.values)**2)
Xval['dR_j1j3'] = np.sqrt(delta_phi(Xval.jet1_phi.values, Xval.jet3_phi.values)**2+ (Xval.jet1_eta.values - Xval.jet3_eta.values)**2)
Xval['dR_j2j3'] = np.sqrt(delta_phi(Xval.jet2_phi.values, Xval.jet3_phi.values)**2+ (Xval.jet2_eta.values - Xval.jet3_eta.values)**2)

Xtrain['dEta_j1j2'] = np.abs(Xtrain.jet1_eta.values - Xtrain.jet2_eta.values)
Xtrain['dEta_j1j3'] = np.abs(Xtrain.jet1_eta.values - Xtrain.jet3_eta.values)
Xtrain['dEta_j2j3'] = np.abs(Xtrain.jet2_eta.values - Xtrain.jet3_eta.values)

Xval['dEta_j1j2'] = np.abs(Xval.jet1_eta.values - Xval.jet2_eta.values)
Xval['dEta_j1j3'] = np.abs(Xval.jet1_eta.values - Xval.jet3_eta.values)
Xval['dEta_j2j3'] = np.abs(Xval.jet2_eta.values - Xval.jet3_eta.values)


Xtrain['dPhi_j1mu'] = np.abs(delta_phi(Xtrain.jet1_phi.values, Xtrain.jet1_muon_phi.values))
Xtrain['dEta_j1mu'] = np.abs(Xtrain.jet1_eta.values - Xtrain.jet1_muon_eta.values)
Xtrain['dR_j1mu'] = np.sqrt(delta_phi(Xtrain.jet1_phi.values, Xtrain.jet1_muon_phi.values)**2 + (Xtrain.jet1_eta.values - Xtrain.jet1_muon_eta.values)**2)
Xtrain['jet1_pt_fraction'] = np.abs(Xtrain.jet1_muon_pt.values/Xtrain.jet1_pt.values)

Xval['dPhi_j1mu'] = np.abs(delta_phi(Xval.jet1_phi.values, Xval.jet1_muon_phi.values))
Xval['dEta_j1mu'] = np.abs(Xval.jet1_eta.values - Xval.jet1_muon_eta.values)
Xval['dR_j1mu'] = np.sqrt(delta_phi(Xval.jet1_phi.values, Xval.jet1_muon_phi.values)**2 + (Xval.jet1_eta.values - Xval.jet1_muon_eta.values)**2)
Xval['jet1_pt_fraction'] = np.abs(Xval.jet1_muon_pt.values/Xval.jet1_pt.values)

# %%
Xtrain = scale_gnn(Xtrain,Xtrain.columns,  scalerName=   "/t3home/gcelotto/ggHbb/GNN/myScaler.pkl" ,fit=True, log=True)
Xval  = scale_gnn(Xval, Xtrain.columns, scalerName=   "/t3home/gcelotto/ggHbb/GNN/myScaler.pkl" ,fit=False, log=True)

# %%
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Data

def build_graph(row, y, w=1.0):
    x = torch.tensor([
        [row.jet1_pt_prime_n,         row.jet1_eta_n,       row.jet1_phi_n,       row.jet1_mass_prime_n,        row.jet1_btagWP_n],
        [row.jet2_pt_prime_n,         row.jet2_eta_n,       row.jet2_phi_n,       row.jet2_mass_prime_n,        row.jet2_btagWP_n],
        [row.jet3_pt_prime_n,         row.jet3_eta_n,       row.jet3_phi_n,       row.jet3_mass_prime_n,        row.jet3_btagWP_n],
        [row.jet1_muon_pt_prime,    row.jet1_muon_eta,  row.jet1_muon_phi,  row.jet1_muon_dxySig,       row.jet1_muon_pfRelIso03_all]
    ], dtype=torch.float)




    edge_index_jetjet = torch.tensor([
        [0, 1, 0, 2, 1, 2],
        [1, 0, 2, 0, 2, 1]
    ], dtype=torch.long)

    edge_attr_jetjet = torch.tensor([
        [row.ptj1j2_e, row.dEta_j1j2_e, row.dPhi_j1j2_e, row.mj1j2_e, row.dR_j1j2_e],
        [row.ptj1j2_e, row.dEta_j1j2_e, row.dPhi_j1j2_e, row.mj1j2_e, row.dR_j1j2_e],
        [row.ptj1j3_e, row.dEta_j1j3_e, row.dPhi_j1j3_e, row.mj1j3_e, row.dR_j1j3_e],
        [row.ptj1j3_e, row.dEta_j1j3_e, row.dPhi_j1j3_e, row.mj1j3_e, row.dR_j1j3_e],
        [row.ptj2j3_e, row.dEta_j2j3_e, row.dPhi_j2j3_e, row.mj2j3_e, row.dR_j2j3_e],
        [row.ptj2j3_e, row.dEta_j2j3_e, row.dPhi_j2j3_e, row.mj2j3_e, row.dR_j2j3_e],
    ], dtype=torch.float)

    # Muon-Jet edges
    edge_index_muonjet = torch.tensor([
        [3, 0],
        [0, 3]
    ], dtype=torch.long)

    edge_attr_muonjet = torch.tensor([
        [row.jet1_pt_fraction, row.dPhi_j1mu, row.dEta_j1mu, row.dR_j1mu],
        [row.jet1_pt_fraction, row.dPhi_j1mu, row.dEta_j1mu, row.dR_j1mu]
    ], dtype=torch.float)



    

    global_features = torch.tensor(
        [[
row.jet1_pt_prime, # log and scaled
row.jet1_eta_prime,
row.jet1_btagWP,
row.jet1_mass_prime, # log and scaled
row.jet2_pt_prime, # log and scaled
row.jet2_eta_prime,
row.jet2_phi_prime,
row.jet2_btagWP,
row.jet2_mass_prime, # log and scaled
row.jet3_pt_prime, # log and scaled
row.jet3_eta_prime,
row.jet3_phi_prime,
row.jet3_btagWP,
row.jet3_mass_prime, # log and scaled
row.jet1_muon_pt_prime, # log and scaled
row.jet1_muon_eta_prime,
row.jet1_muon_phi_prime,
row.jet1_muon_dxySig,
row.dijet_pt_prime, # log and scaled
row.dijet_eta_prime,
row.dijet_phi_prime,
row.dijet_mass, # not log but scaled
row.dijet_dR,
row.dijet_dEta,
row.dijet_twist,
row.dijet_cs,
row.dR_jet3_dijet,
row.dPhi_jet3_dijet,
row.nJets_20,
row.nJets_30,
row.nJets_50,
row.ht_prime # log and scaled
        ]],
        dtype=torch.float
    )

    data= Data(x=x,
                y=torch.tensor(y, dtype=torch.float),
                weight=torch.tensor(w, dtype=torch.float),
                u=global_features)
    data.edge_index_jetjet = edge_index_jetjet
    data.edge_attr_jetjet = edge_attr_jetjet
    data.edge_index_muonjet = edge_index_muonjet
    data.edge_attr_muonjet = edge_attr_muonjet

    data.mjj = torch.tensor(row.dijet_mass, dtype=torch.float)
    data.w = torch.tensor(w, dtype=torch.float)
    type_id = torch.tensor([0, 0, 0, 1], dtype=torch.long)

    data.type_id = type_id
    return data


# %%
N = 1024*1024*64
Xtrain = Xtrain.iloc[:N]
Ytrain = Ytrain[:N]
Xval = Xval.iloc[:N]
Yval = Yval[:N]
genMassTrain = genMassTrain[:N]
genMassVal = genMassVal[:N]
# %%


import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader

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

# =====================
# Example usage
# =====================

# Suppose you already built graphs
# graphs = [build_graph(Xtrain.iloc[i], Ytrain[i]) for i in range(len(Xtrain))]




# %%
# Normalization per node:
cols1 = ['jet1_pt_prime', 'jet2_pt_prime','jet3_pt_prime']
cols2 = ['jet1_eta', 'jet2_eta','jet3_eta']
cols3 = ['jet1_phi', 'jet2_phi', 'jet3_phi']
cols4 = ['jet1_mass_prime', 'jet2_mass_prime','jet3_mass_prime']
cols5 = ['jet1_btagWP', 'jet2_btagWP','jet3_btagWP']

cols6 = ["ptj1j2", "ptj1j3", "ptj2j3"]
cols7 = ["mj1j3", "mj2j3", "mj1j2"]
cols8 =["dEta_j1j2",     "dEta_j1j3",     "dEta_j2j3", ]
cols9 =["dPhi_j1j2",     "dPhi_j1j3",     "dPhi_j2j3", ]
cols10 =["dR_j1j2",     "dR_j1j3",     "dR_j2j3", ]

cols_muons_global = [ 
"jet1_pt_prime", # log done and scaled
"jet1_eta_prime",
"jet1_btagWP",
"jet1_mass_prime", # log done and scaled
"jet2_pt_prime", # log done and scaled
"jet2_eta_prime",
"jet2_phi_prime",
"jet2_btagWP",
"jet2_mass_prime", # log done and scaled
"jet3_pt_prime", # log done and scaled
"jet3_eta_prime",
"jet3_phi_prime",
"jet3_btagWP",
"jet3_mass_prime", # log done and scaled
"jet1_muon_pt_prime", # log done and scaled
"jet1_muon_eta_prime",
"jet1_muon_phi_prime",
"jet1_muon_dxySig",
"dijet_pt_prime",   # log done and scaled
"dijet_eta_prime",
"dijet_phi_prime",
"dijet_mass", # not log but scaled
"dijet_dR",
"dijet_dEta",
"dijet_twist",
"dijet_cs",
"dR_jet3_dijet",
"dPhi_jet3_dijet",
"nJets_20",
"nJets_30",
"nJets_50",
"ht_prime" # log and scaled done
]
# %%
groups = {
    "_n": [cols1, cols2, cols3, cols4, cols5],
    "_e": [cols6, cols7, cols8, cols9, cols10],
}
def normalize_group(Xtrain, Xval, cols, suffix):
    mu = Xtrain[cols].to_numpy().mean()
    sigma = Xtrain[cols].to_numpy().std()
    print("Mean is ", mu)
    Xtrain[[c + suffix for c in cols]] = (Xtrain[cols] - mu) / sigma
    Xval[[c + suffix for c in cols]]   = (Xval[cols]   - mu) / sigma
for suffix, col_lists in groups.items():
    for cols in col_lists:
        normalize_group(Xtrain, Xval, cols, suffix)



for col in cols_muons_global:
    Xval[col] = (Xval[col] - Xtrain[col].values.mean())/Xtrain[col].values.std()
    Xtrain[col] = (Xtrain[col] - Xtrain[col].values.mean())/Xtrain[col].values.std()

# %%
# Cross check
    
import math

for col_group in [cols1, cols2, cols3, cols4, cols5]:
    suffix = "_n"
    assert(math.isclose(Xtrain[[c + suffix for c in col_group]].values.mean() ,0,abs_tol=1e-4))
    assert(math.isclose(Xval[[c + suffix for c in col_group]].values.mean()  ,0,abs_tol=1e-2))
    assert(math.isclose(Xtrain[[c + suffix for c in col_group]].values.std() , 1, abs_tol=1e-4))
    assert(math.isclose(Xval[[c + suffix for c in col_group]].values.std()  , 1, abs_tol=1e-2))

for col_group in [cols6, cols7, cols8, cols9, cols10]:
    suffix = "_e"
    assert(math.isclose(Xtrain[[c + suffix for c in col_group]].values.mean() , 0, abs_tol=1e-4))
    assert(math.isclose(Xval[[c + suffix for c in col_group]].values.mean()  , 0, abs_tol=1e-2))
    assert(math.isclose(Xtrain[[c + suffix for c in col_group]].values.std() , 1, abs_tol=1e-4))
    assert(math.isclose(Xval[[c + suffix for c in col_group]].values.std()  , 1, abs_tol=1e-2))

# %%
graphs_val = [build_graph(Xval.iloc[i], Yval[i], rWval[i]) for i in range(len(Xval))]
# Wrap in InMemoryDataset
dataset = JetGraphDataset(graphs_val)
print("validation start")
# Save to disk
torch.save(dataset, "/t3home/gcelotto/ggHbb/GNN/graphs_val_hetero_hetero.pt")

print("validation saved")

graphs = [build_graph(Xtrain.iloc[i], Ytrain[i], rWtrain[i]) for i in range(len(Xtrain))]
dataset = JetGraphDataset(graphs)

# Save to disk
torch.save(dataset, "/t3home/gcelotto/ggHbb/GNN/graphs_train_hetero_hetero.pt")

loader = DataLoader(graphs, batch_size=1024, shuffle=True)
loader_val = DataLoader(graphs_val, batch_size=1024, shuffle=True)
# %%

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

data = graphs[5]  # one event

G = nx.DiGraph()  # directed graph (matches GNN)

for i in range(data.num_nodes):
    G.add_node(
        i,
        x=data.x[i].cpu().numpy(),
        type=int(data.type_id[i])
    )

edge_index = data.edge_index_jetjet
edge_attr  = data.edge_attr_jetjet

for k in range(edge_index.shape[1]):
    src = int(edge_index[0, k])
    dst = int(edge_index[1, k])
    attr = edge_attr[k].cpu().numpy()

    G.add_edge(src, dst, kind="jet-jet", edge_attr=attr)

edge_index = data.edge_index_muonjet
edge_attr  = data.edge_attr_muonjet

for k in range(edge_index.shape[1]):
    src = int(edge_index[0, k])
    dst = int(edge_index[1, k])
    attr = edge_attr[k].cpu().numpy()

    G.add_edge(src, dst, kind="muon-jet", edge_attr=attr)

fig, ax = plt.subplots(1, 1)

pos = nx.spring_layout(G, seed=42)

node_colors = [
    "tab:blue" if G.nodes[n]["type"] == 0 else "tab:red"
    for n in G.nodes
]

nx.draw(
    G,
    pos,
    ax=ax,
    with_labels=True,
    node_color=node_colors,
    node_size=600,
    arrows=True
)

node_labels = {
    i: f"{np.round(G.nodes[i]['x'], 2)}"
    for i in G.nodes
}

nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, ax=ax)
edge_labels = {
    (u, v): G.edges[u, v]["kind"]
    for u, v in G.edges
}

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
ax.set_title("Jet–Jet and Muon–Jet Graph")
ax.margins(0.2)
plt.tight_layout()
plt.show()

fig.tight_layout()
ax.margins(0.9) 
plt.show()
# %%









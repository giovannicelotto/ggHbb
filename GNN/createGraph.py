import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
# Now, let's define the function to create graphs
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree

def calculate_dijet_mass(jet1, jet2):
    """
    Calculate the dijet mass for two jets given their pt, eta, phi, and mass.
    """
    pt1, eta1, phi1, mass1 = jet1
    pt2, eta2, phi2, mass2 = jet2

    # Jet 4-momentum components
    pT1 = pt1
    pT2 = pt2
    
    # Energy for massless jets: E = pt (simplification)
    E1 = np.sqrt(pt1**2 + mass1**2)  # Energy for jet 1
    E2 = np.sqrt(pt2**2 + mass2**2)  # Energy for jet 2

    # Calculate the momentum components (assuming massless jets, so we use pT)
    p1 = np.array([pT1 * np.cos(phi1), pT1 * np.sin(phi1), pT1 * np.sinh(eta1)])
    p2 = np.array([pT2 * np.cos(phi2), pT2 * np.sin(phi2), pT2 * np.sinh(eta2)])

    # Energy and momentum sum
    E_total = E1 + E2
    p_total = p1 + p2

    # Dijet mass squared: m^2 = (E1 + E2)^2 - |p1 + p2|^2
    dijet_mass_squared = E_total**2 - np.sum(p_total**2)
    
    # To avoid negative square roots (numerical issues), we clamp the result
    dijet_mass = np.sqrt(max(dijet_mass_squared, 0))
    return dijet_mass


def create_graph(event_data, label, k_neighbors=2):
    """
    Create a graph representation for a single event, where each jet is a node,
    and edges are formed based on nearest neighbor search in the (eta, phi) space.
    
    Args:
        event_data (pd.Series): A single row from the DataFrame representing one event.
        label (int): The label for the event (signal=1, background=0).
        k_neighbors (int): Number of nearest neighbors to connect each jet.
    
    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
    """
    nJets = int(event_data['nJets'])
    jet_features = []

    # Loop over the first 3 jets, or up to nJets if there are less than 3 jets
    for i in range(1, min(4, nJets+1)):  # Loop over the first 3 jets or less
        pt = event_data[f'jet{i}_pt']
        eta = event_data[f'jet{i}_eta']
        phi = event_data[f'jet{i}_phi']
        mass = event_data[f'jet{i}_mass']
        jet_features.append([pt, eta, phi, mass])

    jet_features = np.array(jet_features)
    x = torch.tensor(jet_features, dtype=torch.float)
    k_neighbors = int(min(k_neighbors, nJets - 1))
    
    edge_index = []
    edge_attr = []  # Will store the dijet_mass for each edge
    
    if nJets > 1:
        tree = KDTree(jet_features[:, 1:3])  # Use (eta, phi) as the distance metric for neighbors
        for i in range(min(3, nJets)):  # Only loop over the first 3 jets
            _, neighbors = tree.query(jet_features[i, 1:3].reshape(1, -1), k=k_neighbors + 1)  # Reshape for query
            
            # neighbors[0][1:] excludes the first neighbor which is the jet itself
            valid_neighbors = neighbors[0][1:]
            for n in valid_neighbors:
                if n < min(3, nJets):  # Only add valid neighbors within the available jets
                    if i < n:  # Avoid duplicate edges, i.e., if (i, n) is already added, don't add (n, i)
                        edge_index.append((i, int(n)))
                        # Calculate and store the dijet mass for this pair
                        dijet_mass = calculate_dijet_mass(jet_features[i], jet_features[int(n)])
                        edge_attr.append(dijet_mass)
                    
    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # Convert to PyTorch tensor (shape [2, num_edges])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)  # Dijets mass for edges (shape [num_edges, 1])

    # Convert label to tensor
    y = torch.tensor(label, dtype=torch.long)

    # Return the Data object that PyTorch Geometric uses
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

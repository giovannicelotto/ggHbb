def exponential_shape_loss(x, score, lam=0.0077, threshold=0.8, alpha=10.0, gamma=1.0, eta=0.1, c_eff=0.3):
    """
    x: variable of interest (tensor, shape [batch])
    score: NN score output (tensor, shape [batch])
    lam: exponential parameter
    threshold: soft cut on NN score
    alpha: sharpness of sigmoid
    gamma: weight of shape loss
    eta: weight of yield penalty
    c_eff: target effective selection fraction
    """
    # soft weights
    w = torch.sigmoid(alpha * (score - threshold))
    
    # weighted negative log-likelihood for exponential
    nll = -(w * (torch.log(torch.tensor(lam)) - lam * x)).sum() / (w.sum() + 1e-8)
    
    # yield penalty to avoid collapsing weights
    norm_penalty = ((w.mean() - c_eff) ** 2)
    
    # total loss
    loss = gamma * nll + eta * norm_penalty
    return loss
import torch
import torch.nn.functional as F

def soft_dynamic_bin_exponential_loss(x, score, lam=0.0077,
                                      target_fractions=[0.10, 0.01, 0.005, 0.0025],
                                      alpha=20.0, gamma=1.0):
    """
    x: variable of interest (tensor, shape [batch])
    score: NN output (tensor, shape [batch], assumed normalized 0-1)
    lam: exponential parameter
    target_fractions: target background fraction per bin
    alpha: steepness of sigmoid for soft binning
    gamma: weight of shape loss

    Returns:
        loss: scalar differentiable loss
    """
    batch_size = x.shape[0]
    n_bins = len(target_fractions)
    
    # Sort scores descending
    score_sorted, _ = torch.sort(score, descending=True)

    # Compute cumulative target fractions to define bin edges
    cumulative = torch.cumsum(torch.tensor(target_fractions, device=score.device), dim=0)

    # Compute thresholds per bin (top fraction of events)
    bin_edges = []
    for c in cumulative:
        idx = int(c * batch_size) - 1
        idx = max(idx, 0)  # avoid negative index
        bin_edges.append(score_sorted[idx].item())
    
    # Add minimum and maximum to make exclusive bins
    bin_edges = [score_sorted[-1].item() - 1e-6] + bin_edges  # extend lower edge
    #print("bin_edges for current batch are ", bin_edges)
    # bin_edges: length n_bins +1

    total_loss = 0.0

    for b in range(n_bins):
        lower = bin_edges[b]
        upper = bin_edges[b+1]

        # Soft exclusive bin weight
        w_bin = torch.sigmoid(alpha * (score - lower)) - torch.sigmoid(alpha * (score - upper))

        # Weighted exponential NLL
        w_sum = w_bin.sum() + 1e-8
        nll = -(w_bin * (torch.log(torch.tensor(lam)) - lam * x)).sum() / w_sum
        total_loss += nll

    # Scale by gamma
    loss = gamma * total_loss
    return loss

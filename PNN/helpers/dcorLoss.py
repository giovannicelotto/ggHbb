import torch
import torch.nn as nn
import torch.nn.init as init

# MultiGPU per SingleNode
import torch.multiprocessing as mp #wrapper around python mp
from torch.utils.data.distributed import DistributedSampler #takes input data and distributes accross gpu
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group #initialize and destroy the distributed process group
import os
import random
import numpy as np
#import torch.optim as optim
def distance_corr(var_1,var_2,normedweight,power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    # var_1 is first reshaped as a line vector
    # xx is matrix where each row is var_1
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    # yy is matrix where each column is var_2
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()
    # amat will be zero diagonally, symmetric (because of absolute value), and is a matrix of differences betwenn all the elements of var_1

    # same is done for var_2 and bmat
    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    # Compute the mean in each row after multiplying by normed weights. Then the mean of row will be repeated to match the shape of the matrix (see after : .repeat(len(var_1),1).view(len(var_1),len(var_1)))
    amatavg = torch.mean(amat*normedweight,dim=1)
    # Subtract to amat the mean_row, mean_column, and add the grand mean
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    
    return dCorr

def pearsonR(var1, var2, w):
    x_mean = torch.mean(var1)
    y_mean = torch.mean(var2)

    x_diff = var1 - x_mean
    y_diff = var2 - y_mean

    numerator = torch.sum(x_diff * y_diff)
    denominator = torch.sqrt(torch.sum(x_diff ** 2)) * torch.sqrt(torch.sum(y_diff ** 2))

    pearson_corr = numerator / (denominator + 1e-8)  # Adding epsilon for numerical stability

    return abs(pearson_corr )

# Function for MP
def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For single-GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Neural Network Classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, nNodes=[128, 64, 32]):
        """
        Initialize the classifier model.
        
        Args:
            input_dim (int): Number of input features.
            nNodes (list or tuple): Number of nodes in each hidden layer.
                                     Example: [128, 64, 32]
        """
        super(Classifier, self).__init__()
        
        layers = []
        current_dim = input_dim  # Start with input dimension
        n = nNodes[0]
        layers.append(nn.Linear(current_dim, n))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(n))
        current_dim = n
        #layers.append(nn.Dropout(dropout_prob))
        for n in nNodes[1:]:
            layers.append(nn.Linear(current_dim, n))
            layers.append(nn.ReLU())
            # Update current dimension to the output of this layer
            current_dim = n  
        
        # Add the final output layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())

        # Combine all layers into a Sequential module
        self.fc = nn.Sequential(*layers)

        # Apply weight initialization
        self.apply(self._initialize_weights)
        
    
    
    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            #init.xavier_normal_(layer.weight)
            # Kaiming Initialization for layers with ReLU activations
            init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            # Initialize biases to 0
            init.constant_(layer.bias, 0)

            # Special treatment for the output layer (Sigmoid output layer)
            if layer.out_features == 1:
                init.xavier_normal_(layer.weight)  
                init.constant_(layer.bias, 0)      

# Theory
# initializing all weights to zero in neural networks is largely inadvisable due to the critical issue of symmetry breaking.
# However, zero initialization for biases is generally acceptable as it does not contribute to the symmetry problem in the same way weights do. Biases can start from zero since their primary role is to provide an adjustable threshold for neuron activation rather than to diversify learning paths.
#                The batch normalization layer has two learnable parameters:
#
#weight (also called gamma): This is the scaling factor applied to the normalized output.
#bias (also called beta): This is the shifting factor added after normalization.
#Typically, the initialization for these parameters is:
#
#weight (gamma): Initialized to 1 (to preserve the identity transformation initially).
#bias (beta): Initialized to 0 (no shift initially).
#PyTorch's nn.BatchNorm1d layers are automatically initialized to these defaults, so you don't necessarily need to manually initialize them unless you have a specific reason to do so.

    # this is autocalled for predictions. apply the model (self) to x
    def forward(self, x):
        return self.fc(x)



def kolmogorov_smirnov_distance_weighted(p, scores, threshold, k):
    """
    Compute the Kolmogorov-Smirnov distance between two 1D distributions with sigmoid-based weights.
    """
    # Sort both distributions
    p_sorted, p_idx = torch.sort(p)
    sorted_scores = scores[p_idx]
    
    sigmoid_weights_high = torch.sigmoid(k * (sorted_scores - threshold))  # High value of NN
    sigmoid_weights_low = torch.sigmoid(k * (threshold - sorted_scores))
    
    
    # Normalize weights
    weights_high = sigmoid_weights_high / sigmoid_weights_high.sum()
    weights_low = sigmoid_weights_low / sigmoid_weights_low.sum()
    
    # Compute the weighted CDFs by summing weights cumulatively
    cdf_high = torch.cumsum(weights_high, dim=0)
    cdf_low = torch.cumsum(weights_low, dim=0)
    
    # Compute the absolute differences between the weighted CDFs
    ks_distance = torch.max(torch.abs(cdf_high - cdf_low))
    
    return ks_distance
# Example data
#p_samples = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Samples from distribution P
#scores = torch.tensor([0.1, 0.4, 0.6, 0.3])     # NN scores associated with the samples
#threshold = 0.5  # Threshold for separating high and low
#
## Compute the KS distance with weighted CDFs
#ks_distance = kolmogorov_smirnov_distance_weighted(p_samples, scores, threshold, k=10000)
#print(f"Weighted Kolmogorov-Smirnov Distance: {ks_distance.item()}")
def bimodality(nnscore):
    # nnscore is expected to be a PyTorch tensor
    n = len(nnscore)
    
    # Compute skewness and kurtosis manually to support gradient computation
    mean = torch.mean(nnscore)
    std = torch.std(nnscore, unbiased=True)
    
    # Skewness (gamma1)
    gamma1 = torch.mean(((nnscore - mean) / std) ** 3)
    
    # Kurtosis (gamma2) using Fisher's definition (excess kurtosis)
    gamma2 = torch.mean(((nnscore - mean) / std) ** 4) - 3
    
    # Calculate the bimodality coefficient
    bimodality = (gamma1**2 + 1) / (gamma2 + 3 * ((n - 1)**2) / ((n - 2) * (n - 3)))
    
    return bimodality


def varianceDcor(dCor):
    """
        Compute the variance of the tensor
    """
    variance = torch.var(dCor)
    return variance
    



# Idea for future losses
#import numpy as np
#import torch
#from scipy.stats import gaussian_kde
#
#def smoothness_measure(sample, bandwidth=0.1, num_points=100):
#    """
#    Computes a differentiable smoothness measure based on the L2 norm of the second derivative.
#
#    Parameters:
#    - sample (array-like): 1D array of samples from the distribution.
#    - bandwidth (float): Bandwidth for KDE.
#    - num_points (int): Number of points to evaluate smoothness.
#
#    Returns:
#    - smoothness (torch.Tensor): Scalar smoothness measure.
#    """
#    sample = np.array(sample)
#    
#    # Estimate PDF using Kernel Density Estimation (KDE)
#    kde = gaussian_kde(sample, bw_method=bandwidth)
#    
#    # Create evaluation points
#    x_eval = np.linspace(sample.min(), sample.max(), num_points)
#    pdf_values = kde(x_eval)
#    
#    # Convert to torch tensor for differentiation
#    x_eval_torch = torch.tensor(x_eval, dtype=torch.float32, requires_grad=True)
#    pdf_values_torch = torch.tensor(pdf_values, dtype=torch.float32)
#    
#    # Compute first and second derivatives using autograd
#    first_derivative = torch.autograd.grad(pdf_values_torch, x_eval_torch, create_graph=True)[0]
#    second_derivative = torch.autograd.grad(first_derivative, x_eval_torch, create_graph=True)[0]
#
#    # Compute L2 norm of the second derivative
#    smoothness = torch.norm(second_derivative, p=2)
#
#    return smoothness
#
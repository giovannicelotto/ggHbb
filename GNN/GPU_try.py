import torch
print("Torch version ", torch.__version__)       # esempio: 2.1.0
print("Cuda version", torch.version.cuda) 
torch.backends.cudnn.benchmark = True
gpuFlag=True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)
print("Cuda is Available", torch.cuda.is_available())
print("Device count", torch.cuda.device_count())
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from helpers.getParams import getParams
from datetime import datetime
import argparse
# Torch
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader, TensorDataset

# MultiGPU per SingleNode
import torch.multiprocessing as mp #wrapper around python mp
from torch.utils.data.distributed import DistributedSampler #takes input data and distributes accross gpu
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group #initialize and destroy the distributed process group
import os
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
import torch.multiprocessing as mp
if __name__ == '__main__':
    # Get current month and day
    current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
    # %%
    gpuFlag=True if torch.cuda.is_available() else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define folder of input and output. Create the folders if not existing
    hp = getParams()
    parser = argparse.ArgumentParser(description="Script.")
    #### Define arguments
    parser.add_argument("-l", "--lambda_dcor", type=float, help="lambda for penalty term", default=None)
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=None)
    parser.add_argument("-s", "--size", type=int, help="Number of events to crop training dataset", default=None)
    parser.add_argument("-bs", "--batch_size", type=int, help="Number of eventsper batch size", default=None)
    parser.add_argument("-node", "--node", type=int, help="nodes of single layer in case of one layer for simple nn", default=None)
    # %%
    try:
        args = parser.parse_args()
        if args.lambda_dcor is not None:
            hp["lambda_dcor"] = args.lambda_dcor 
            print("lambda_dcor changed to ", hp["lambda_dcor"])
        if args.epochs is not None:
            hp["epochs"] = args.epochs 
            print("N epochs to ", hp["epochs"])
        if args.batch_size is not None:
            hp["batch_size"] = int(args.batch_size )
            print("N batch_size to ", hp["batch_size"])
        if args.node is not None:
            hp["nNodes"] = [args.node]
        if args.size is not None:
            hp["size"]=args.size

    except:
        print("-"*40)
        print("No arguments provided for lambda!")
        print("lambda_dcor changed to ", hp["lambda_dcor"])
        hp["lambda_dcor"] = 5. 
        hp["size"] = int(1e9)
        print(hp)
        print("interactive mode")
        print("-"*40)



    # Number of GPUs you're using
    world_size = torch.cuda.device_count()
    # Number of epochs and how often to save
    total_epochs = 100
    save_every = 10
    # Spawn the processes for distributed training
    from DoubleDiscoPNNTorchGPU_MultiNode import main2
    
    mp.spawn(main2, args=(world_size, device, current_date, hp), nprocs=world_size)

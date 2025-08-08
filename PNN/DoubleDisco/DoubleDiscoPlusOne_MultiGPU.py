# Generic import
import numpy as np
import sys
import time
from datetime import datetime
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
sys.path.append("/t3home/gcelotto/ggHbb/PNN")

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributed import barrier

# PNN helpers
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams, print_params
from helpers.loadSaved import loadXYWrWSaved
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale
from helpers.dcorLoss import *

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

'''
Code to train two NNs to produce scores which are independent.
Starting from same input features two discriminating scores are produced which can be used for ABCD method for background estimation
Arguments:
-l              : lambda_dcor. Penalization of distance correlation
-c              : lambda_closure. Penalization of closure of ABCD
--epochs        : number of epochs
--size          : size of dataset (ignore it if you want to use all dataset available for training)
--batch_size    : batch size
--nodes         : number of nodes written as comma separated e.g. 128,64,32


'''








def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12356")
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        nn1: torch.nn.Module,
        nn2: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer1: torch.optim.Optimizer,
        optimizer2: torch.optim.Optimizer,
        gpu_id: int,
        criterion,
        mass_bins,
        #save_every: int,
    ) -> None:
        self.rank = gpu_id
        self.nn1 = nn1.to(gpu_id)
        self.nn2 = nn2.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        #self.save_every = save_every
        self.nn1 = DDP(self.nn1, device_ids=[gpu_id])
        self.nn2 = DDP(self.nn2, device_ids=[gpu_id])
        self.criterion = criterion
        self.mass_bins = mass_bins


    #def _save_checkpoint(self, epoch):
    #    ckp = self.model.module.state_dict()
    #    PATH = "checkpoint.pt"
    #    torch.save(ckp, PATH)
    #    print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, hp: dict, outFolder: str, save: bool, modelName: str):
        best_model_weights1, best_model_weights2 = None, None  
        best_val_loss = float('inf')
        
        train_classifier_loss_history   = []
        train_loss_history              = []
        train_dcor_loss_history         = []    
        train_closure_loss_history      = []
        val_classifier_loss_history     = []
        val_loss_history                = []
        val_dcor_loss_history           = []    
        val_closure_loss_history        = []

        batch_start_time = time.time()
        final_lambda_dcor = hp["lambda_dcor"]  # Store the final value
        final_lambda_closure = hp["lambda_closure"]  # Store the final value
        hp["lambda_dcor"] = 0  # Start with 0
        hp["lambda_closure"] = 0  # Start with 0
        hp["lambda_b2"] = 0  
        warmup_epochs = 20
        step_size = 2  # Increase every step_size epochs
        for epoch in range(hp["epochs"]):
            if (epoch % step_size == 0) & (epoch <= warmup_epochs):
                hp["lambda_dcor"] = final_lambda_dcor * (epoch / warmup_epochs)
                hp["lambda_closure"] = final_lambda_closure * (epoch / warmup_epochs)
            if epoch > warmup_epochs:
                hp["lambda_dcor"] = final_lambda_dcor
                hp["lambda_closure"] = final_lambda_closure
            self.train_data.sampler.set_epoch(epoch)
            self.nn1.train()
            self.nn2.train()
            
            # Float for keeping track of Loss
            total_trainloss = 0.0
            total_trainclassifier_loss = 0.0
            total_traindcor_loss = 0.0
            total_trainclosure_loss = 0.0

            

        # ______________________________________________________________________________
        # ______________________________________________________________________________
        # ______________________  TRAINING   PHASE  ____________________________________
        # ______________________________________________________________________________
        # ______________________________________________________________________________
            for batch in self.train_data:
                
                X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch, jet2_btagTight_batch = batch


                # Reset the gradients of all optimized torch.Tensor. Otherwise one step is cumulative sum of gradients of previous steps
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                predictions1 = self.nn1(X_batch)
                predictions2 = self.nn2(X_batch)
                if (predictions1.min().item()<0) | (predictions1.max().item()>1) | (predictions2.min().item()<0) | (predictions2.max().item()>1):
                    print("Error on predictions :\nMin Prediction : %.2f %.2f \nMax Prediction : %.2f %.2f"%(predictions1.min().item(), predictions2.min().item(), predictions1.max().item(), predictions2.max().item()))
                raw_loss1 = self.criterion(predictions1, Y_batch)
                raw_loss2 = self.criterion(predictions2, Y_batch)
                classifier_loss1 = (raw_loss1 * W_batch).mean()
                classifier_loss2 = (raw_loss2 * W_batch).mean()


                W_batch = torch.ones([len(W_batch), 1], device=self.rank)
                dCorr_total = 0.
                closure_total = 0.
                dCorr_b2_total = 0.
                for low, high in zip(self.mass_bins[:-1], self.mass_bins[1:]):
                    # Mask for the current mass bin and bkg events only
                    bin_mask = (dijetMass_batch >= low) & (dijetMass_batch < high)  & (mask_batch)
                    assert bin_mask.size(0) == predictions1.size(0), f"Mismatch: bin_mask size {bin_mask.size(0)} vs predictions1 size {predictions1.size(0)}"
                    
                    #if (bin_mask.sum().item()==0):
                    #    print("No elements in this bin!")
                    #    continue
                                                # Apply bin-specific mask
                    bin_predictions1 = predictions1[bin_mask]
                    bin_predictions2 = predictions2[bin_mask]
                    bin_weights = W_batch[bin_mask]
                    bin_jet2_btagTight_batch = jet2_btagTight_batch[bin_mask]

                
                    if hp["lambda_closure"]!=0:
                        closure_bin = closure(bin_predictions1, bin_predictions2, bin_weights, symmetrize=True, n_events_min=10)
                        closure_total += closure_bin
                    # Compute dCorr for the current mass bin
                    if hp["lambda_dcor"]!=0:
                        dCorr_bin = distance_corr(bin_predictions1, bin_predictions2, bin_weights)
                        dCorr_total += dCorr_bin 
                    if hp["lambda_b2"]!=0:
                        dCorr_bin =  distance_corr(bin_predictions1, bin_jet2_btagTight_batch, bin_weights) 
                        dCorr_bin += distance_corr(bin_predictions2, bin_jet2_btagTight_batch, bin_weights)
                        dCorr_b2_total += dCorr_bin

                loss = classifier_loss1 +classifier_loss2 
                if hp["lambda_closure"]!=0:
                    loss += hp["lambda_closure"] * closure_total/(len(self.mass_bins) -1)
                if hp["lambda_dcor"]!=0:
                    loss +=  hp["lambda_dcor"] * dCorr_total/(len(self.mass_bins) -1)
                if hp["lambda_b2"]!=0:
                    loss +=  hp["lambda_b2"] * dCorr_b2_total/(len(self.mass_bins) -1)
                
                loss.backward()

                self.optimizer1.step()
                self.optimizer2.step()
                #scheduler1.step()
                #scheduler2.step()
                #for param_group in optimizer1.param_groups:
                #    param_group['lr'] = max(param_group['lr'], hp['min_learning_rate'])  
                #
                #for param_group in optimizer2.param_groups:
                #    param_group['lr'] = max(param_group['lr'], hp['min_learning_rate'])

                total_trainloss += loss.item()
                total_trainclassifier_loss += classifier_loss1.item() + classifier_loss2.item()
                if hp["lambda_closure"]!=0:
                    total_trainclosure_loss += closure_total.item()/(len(self.mass_bins) -1)
                if hp["lambda_dcor"]!=0:
                    total_traindcor_loss += dCorr_total.item()/(len(self.mass_bins) -1)


        # ______________________________________________________________________________
        # ______________________________________________________________________________
        # ______________________  VALIDATION PHASE  ____________________________________
        # ______________________________________________________________________________
        # ______________________________________________________________________________




            if epoch % 1 == 0: 
                self.nn1.eval()
                self.nn2.eval()
                total_val_loss = 0.0
                total_val_classifier_loss = 0.0
                total_val_dcor_loss = 0.0
                total_val_closure_loss = 0.
                

                with torch.no_grad():
                    for batch in self.val_data:
                        X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch, jet2_btagTight_batch = batch
                        predictions1 = self.nn1(X_batch)
                        predictions2 = self.nn2(X_batch)
                        

                        raw_loss1 = self.criterion(predictions1, Y_batch)
                        raw_loss2 = self.criterion(predictions2, Y_batch)
                    # Apply weights manually
                        classifier_loss1 = (raw_loss1 * W_batch).mean()
                        classifier_loss2 = (raw_loss2 * W_batch).mean()

                        # If there are any remaining entries after filtering, calculate dcor
                        W_batch = torch.ones([len(W_batch), 1], device=self.rank)
                        dCorr_total = 0.
                        closure_total = 0.
                        dCorr_b2_total = 0.
                        for low, high in zip(self.mass_bins[:-1], self.mass_bins[1:]):
                            # Mask for the current mass bin
                            bin_mask = (dijetMass_batch >= low) & (dijetMass_batch < high) & (mask_batch)

                            # Apply bin-specific mask
                            bin_predictions1 = predictions1[bin_mask]
                            bin_predictions2 = predictions2[bin_mask]
                            bin_jet2_btagTight_batch = jet2_btagTight_batch[bin_mask]
                            bin_weights = W_batch[bin_mask]

                        
                            if hp["lambda_closure"]!=0:
                                closure_bin = closure(bin_predictions1, bin_predictions2, bin_weights,symmetrize=True,n_events_min=10)
                                closure_total += closure_bin
                            # Compute dCorr for the current mass bin
                            if hp["lambda_dcor"]!=0:
                                dCorr_bin = distance_corr(bin_predictions1, bin_predictions2, bin_weights)
                                dCorr_total += dCorr_bin 
                            if hp["lambda_b2"]!=0:
                                dCorr_bin =  distance_corr(bin_predictions1, bin_jet2_btagTight_batch, bin_weights)
                                dCorr_bin += distance_corr(bin_predictions2, bin_jet2_btagTight_batch, bin_weights)
                                dCorr_b2_total += dCorr_bin

                        loss = classifier_loss1 +classifier_loss2 
                        if hp["lambda_closure"]!=0:
                            loss += hp["lambda_closure"] * closure_total/(len(self.mass_bins) -1)
                        if hp["lambda_dcor"]!=0:
                            loss +=  hp["lambda_dcor"] * dCorr_total/(len(self.mass_bins) -1)
                        if hp["lambda_b2"]!=0:
                            loss +=  hp["lambda_b2"] * dCorr_b2_total/(len(self.mass_bins) -1)
                        
                        total_val_loss += loss.item()
                        total_val_classifier_loss += classifier_loss1.item() + classifier_loss2.item()
                        if hp["lambda_closure"]!=0:
                            total_val_closure_loss += closure_total.item()/(len(self.mass_bins) -1)
                        if hp["lambda_dcor"]!=0:
                            total_val_dcor_loss += dCorr_total.item()/(len(self.mass_bins) -1)


            # Calculate average losses (average over batches)
            #print("len(self.train_data)", len(self.train_data))
            #print("len(self.val_data)", len(self.val_data))
            avg_trainloss = total_trainloss / len(self.train_data)
            avg_train_classifier_loss = total_trainclassifier_loss / len(self.train_data)
            avg_traindcor_loss = total_traindcor_loss / len(self.train_data)
            avg_trainclosure_loss = total_trainclosure_loss / len(self.train_data)

            avg_val_loss = total_val_loss / len(self.val_data)
            avg_val_classifier_loss = total_val_classifier_loss / len(self.val_data)
            avg_val_dcor_loss = total_val_dcor_loss / len(self.val_data)
            avg_valclosure_loss = total_val_closure_loss / len(self.val_data)
            
            train_loss_history.append(avg_trainloss)
            train_classifier_loss_history.append(avg_train_classifier_loss)
            train_dcor_loss_history.append(avg_traindcor_loss)
            train_closure_loss_history.append(avg_trainclosure_loss)
            
            val_loss_history.append(avg_val_loss)
            val_classifier_loss_history.append(avg_val_classifier_loss)
            val_dcor_loss_history.append(avg_val_dcor_loss)
            val_closure_loss_history.append(avg_valclosure_loss)
            # Print losses
            if self.rank==0:
                print(f"Epoch [{epoch+1}/{hp['epochs']}], "
                  f"Train Loss: {avg_trainloss:.4f}, Classifier Loss: {avg_train_classifier_loss:.4f}, dCor Loss: {avg_traindcor_loss:.8f}, clos Loss: {avg_trainclosure_loss:.8f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Classifier Loss: {avg_val_classifier_loss:.4f}, Val dCor Loss: {avg_val_dcor_loss:.8f}, clos Loss: {avg_valclosure_loss:.8f}",
                  flush=(epoch % 10 == 0))
                

            # CheckPoints
            if (epoch%10==0):
                if (self.rank==0) & (save):
                    torch.save(self.nn1.state_dict(), outFolder + "/model/nn1_e%d.pth"%epoch)
                    torch.save(self.nn2.state_dict(), outFolder + "/model/nn2_e%d.pth"%epoch)

                if save:
                    np.save(outFolder + "/model/train_loss_history_rank%d.npy"%self.rank, train_loss_history)
                    np.save(outFolder + "/model/val_loss_history_rank%d.npy"%self.rank, val_loss_history)
                    np.save(outFolder + "/model/train_classifier_loss_history_rank%d.npy"%self.rank, train_classifier_loss_history)
                    np.save(outFolder + "/model/val_classifier_loss_history_rank%d.npy"%self.rank, val_classifier_loss_history)
                    np.save(outFolder + "/model/train_dcor_loss_history_rank%d.npy"%self.rank, train_dcor_loss_history)
                    np.save(outFolder + "/model/val_dcor_loss_history_rank%d.npy"%self.rank, val_dcor_loss_history)
                    np.save(outFolder + "/model/train_closure_loss_history_rank%d.npy"%self.rank, train_closure_loss_history)
                    np.save(outFolder + "/model/val_closure_loss_history_rank%d.npy"%self.rank, val_closure_loss_history)

            
            # No early stop for GPU
            #if avg_val_loss < best_val_loss:
            #    best_val_loss = avg_val_loss
            #    patience_counter = 0  # reset patience if validation loss improves
            #    best_model_weights1 = self.nn1.state_dict()
            #    best_model_weights2 = self.nn2.state_dict()
            #    best_epoch= epoch
            #else:
            #    patience_counter += 1

            #if patience_counter >= hp["patienceES"]:
            #    print("Early stopping triggered.")
            #    break

            if (self.rank == 0) & (epoch%10==0):
                batch_total_time = time.time() - batch_start_time
                print(f"Epochs time: {batch_total_time :.4f} sec")
                batch_start_time = time.time()
            
            
        # ****
        # End of Training
        # ****
        print("End of training")
        barrier()

        if self.rank==0:
            # Restore best weights in case of EarlyStopping
            #if best_model_weights1 is not None:
            #    self.nn1.load_state_dict(best_model_weights1)
            #    self.nn2.load_state_dict(best_model_weights2)
            #    if save:
            #        np.save(outFolder + "/model/epoch.npy", np.array([best_epoch]))
            #    print("Restored the best model weights.")
            # Save model weights
            if save:
                torch.save(self.nn1.state_dict(), outFolder + "/model/nn1.pth")
                torch.save(self.nn2.state_dict(), outFolder + "/model/nn2.pth")
            else:
                torch.save(self.nn1.state_dict(),  "/scratch/nn1_%s.pth"%modelName)
                torch.save(self.nn2.state_dict(),  "/scratch/nn2_%s.pth"%modelName)
            print("saved the models")

        if save:
            np.save(outFolder + "/model/train_loss_history_rank%d.npy"%self.rank, train_loss_history)
            np.save(outFolder + "/model/val_loss_history_rank%d.npy"%self.rank, val_loss_history)
            np.save(outFolder + "/model/train_classifier_loss_history_rank%d.npy"%self.rank, train_classifier_loss_history)
            np.save(outFolder + "/model/val_classifier_loss_history_rank%d.npy"%self.rank, val_classifier_loss_history)
            np.save(outFolder + "/model/train_dcor_loss_history_rank%d.npy"%self.rank, train_dcor_loss_history)
            np.save(outFolder + "/model/val_dcor_loss_history_rank%d.npy"%self.rank, val_dcor_loss_history)

            #self._run_epoch(epoch)
            #if self.rank == 0 and epoch % self.save_every == 0:
            #    self._save_checkpoint(epoch)

    
def main(rank: int, world_size: int, hp: dict,
         X_train, Y_train, W_train, rW_train, genMass_train, dijetMass_train, jet2_btagTight_train,
         X_val, Y_val, W_val, rW_val, genMass_val, dijetMass_val, jet2_btagTight_val,
         inFolder, outFolder, inputSubFolder, featuresForTraining, mass_bins, save, modelName):
    

    
    # Assign each process to a GPU
    ddp_setup(rank, world_size)


    # Convert train and test into tensors in the corresponding GPUs
    # Change use rW_train and rW_val
    X_trainTensor = torch.tensor(X_train[featuresForTraining].values, dtype=torch.float32, device=rank)
    Y_trainTensor = torch.tensor(Y_train, dtype=torch.float, device=rank).unsqueeze(1)
    W_trainTensor = torch.tensor(rW_train, dtype=torch.float32, device=rank).unsqueeze(1)
    dijetMass_train_tensor = torch.tensor(dijetMass_train, dtype=torch.float32, device=rank).unsqueeze(1)
    jet2_btagTight_train_tensor = torch.tensor(jet2_btagTight_train, dtype=torch.float32, device=rank).unsqueeze(1)


    X_val_tensor = torch.tensor(X_val[featuresForTraining].values, dtype=torch.float32, device=rank)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float, device=rank).unsqueeze(1)
    W_val_tensor = torch.tensor(rW_val, dtype=torch.float32, device=rank).unsqueeze(1)
    dijetMass_val_tensor = torch.tensor(dijetMass_val, dtype=torch.float32, device=rank).unsqueeze(1)
    jet2_btagTight_val_tensor = torch.tensor(jet2_btagTight_val, dtype=torch.float32, device=rank).unsqueeze(1)

    train_masks = (Y_trainTensor < 0.5).to(rank)
    val_masks = (Y_val_tensor < 0.5).to(rank)


    # Create TorchDataset
    traindataset = TensorDataset(
    X_trainTensor.to(rank),
    Y_trainTensor.to(rank),
    W_trainTensor.to(rank),
    dijetMass_train_tensor.to(rank),
    train_masks.to(rank),
    jet2_btagTight_train_tensor.to(rank)
    )

    val_dataset = TensorDataset(
        X_val_tensor.to(rank),
        Y_val_tensor.to(rank),
        W_val_tensor.to(rank),
        dijetMass_val_tensor.to(rank),
        val_masks.to(rank),
        jet2_btagTight_val_tensor.to(rank)
    )


    hp["batch_size"] = hp["batch_size"] if hp["batch_size"]*world_size<len(X_train) else len(X_train)//world_size
    hp["val_batch_size"] = hp["batch_size"] if hp["batch_size"]*world_size<len(X_val) else len(X_val)//world_size
    print("Train Batch size", hp["batch_size"])
    print("Val Batch size", hp["val_batch_size"])
    # DataLoader(drop_last=True, sampler=DS(drop_last=False)) rules the total events e.g. size=850 and bs =100 8 GPUs. each GPU will see 1 batch with 100 events and 50 events epoch by eopch are not used.
    # DataLoader(drop_last=False, sampler=DS(drop_last=True)) . example before. each gpu will have 2 batches one with 100 and one with 6 8*100+8*6 = 848 and the additional 2 events are dropped
    # Given the dependence of disco wrt number of samples set all of them to True (equivalent to 1st=True and 2nd=False)
    # Check that samples not used change time by time. Epoch by epoch the events not used change! Because of shuffle=True inside DistributedSampler (default value)
    # Shuffle of datalaoder is mutually exclusive with sampler option. Inside DistributedSampler shuffle is True by default
    print("train data loader")
    traindataloader = DataLoader(traindataset, batch_size=hp["batch_size"], shuffle=False, drop_last=True, pin_memory=False, sampler=DistributedSampler(traindataset, num_replicas=world_size, rank=rank, drop_last=True))
    print("val data loader")
    val_dataloader = DataLoader(val_dataset, batch_size=hp["val_batch_size"], shuffle=False, drop_last=True, pin_memory=False, sampler=DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, drop_last=True))

    set_seed(1)
    nn1 = Classifier(input_dim=X_train[featuresForTraining].shape[1], nNodes=hp["nNodes"], dropout_prob=0.1)
    set_seed(2)
    nn2 = Classifier(input_dim=X_train[featuresForTraining].shape[1], nNodes=hp["nNodes"], dropout_prob=0.1)
    nn1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(nn1)
    nn2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(nn2)
    nn1.to(rank)
    nn2.to(rank)

    criterion = nn.BCELoss(reduction='none')
    optimizer1 = optim.Adam(nn1.parameters(), lr=hp["learning_rate"])
    optimizer2 = optim.Adam(nn2.parameters(), lr=hp["learning_rate"])
    print("Trainer definition")
    # Define a trainer class
    trainer = Trainer(nn1=nn1, nn2=nn2, train_data=traindataloader, val_data=val_dataloader,
                      optimizer1=optimizer1, optimizer2=optimizer2, gpu_id=rank, criterion=criterion,
                      mass_bins=mass_bins)
    print("Train start")
    trainer.train(hp, outFolder=outFolder, save=save, modelName=modelName)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Script.")
    parser.add_argument("-l", "--lambda_dcor", type=str, help="lambda for penalty term", default=300)
    parser.add_argument("-c", "--lambda_closure", type=str, help="lambda for penalty term", default=500)
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=5000)
    parser.add_argument("-s", "--size", type=int, help="Number of events to crop training dataset", default=3000000)
    parser.add_argument("-bs", "--batch_size", type=int, help="Number of eventsper batch size", default=32768)
    parser.add_argument("-lr", "--learningRate", type=str, help="learningRate", default=0.001)
    parser.add_argument("-n", "--nodes",type=lambda s: [int(item) for item in s.split(',')],  # Convert comma-separated string to list of ints
                        help="List of nodes per layer (e.g., 128,64,32 for a 3-layer NN)",default=None
)
    parser.add_argument("-sampling", "--sampling", type=int, help="input data from sampling signal method (True) or equal size of sample (false)", default=False)
    parser.add_argument("-b", "--boosted", type=int, help="pt 0-100 100-160 160-Inf", default=0)
    parser.add_argument("-dt", "--datataking", type=str, help="1A or 1D", default='1D')
    parser.add_argument("-bin_center", "--bin_center_bool", type=int, help="Use bin center as feature", default=True)
    parser.add_argument("-mH", "--massHypo", type=int, help="Use closest dijet mass discrete value as feature", default=False)
    parser.add_argument("-save", "--saveResults", type=int, help="saveResults (False in case of HP tuning)", default=True)
    parser.add_argument("-modelName", "--modelName", type=str, help="modelName (Only for HP tuning)", default='None')

    args = parser.parse_args()
    
    #hp=getParams()
    hp={}
    hp["lambda_dcor"] = float(args.lambda_dcor )
    hp["lambda_closure"] = float(args.lambda_closure)
    hp["epochs"] = args.epochs 
    hp["batch_size"] = int(args.batch_size )
    hp["size"]=args.size
    hp["sampling"] = args.sampling
    hp["bin_center_bool"] = args.bin_center_bool
    hp["massHypo"] = args.massHypo
    hp["nNodes"] = args.nodes
    hp["learning_rate"] = float(args.learningRate)

    print_params(hp)

    start_time = time.time()
    current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'

    # Define infolder and outfolder based on lambda_dcor, saveResults, sampling, boosted
    inFolder, outFolder = getInfolderOutfolder(name = "%s_%s"%(current_date, str(hp["lambda_dcor"]).replace('.', 'p')), suffixResults="DoubleDisco", createFolder=True if args.saveResults else False)
    inputSubFolder = 'data' if not hp["sampling"] else 'data_sampling'
    inputSubFolder = inputSubFolder+"_pt%d_%s"%(args.boosted, args.datataking)
    print("Input subfolder", inputSubFolder)

    # Define features to read and to train the NN (based on bin_center_bool)
    featuresForTraining, columnsToRead = getFeatures(outFolder=outFolder)
    if hp['bin_center_bool']:
        featuresForTraining=featuresForTraining+['bin_center']
    np.save(outFolder+"/featuresForTraining.npy", featuresForTraining)

    X_train, X_val, Y_train, Y_val, W_train, W_val, rW_train, rW_val, genMass_train, genMass_val = loadXYWrWSaved(inFolder=inFolder+"/%s"%inputSubFolder, isTest=False)

    dijetMass_train = np.array(X_train.dijet_mass.values)
    dijetMass_val = np.array(X_val.dijet_mass.values)

    jet2_btagTight_train = np.array(X_train.jet2_btagTight.values)
    jet2_btagTight_val = np.array(X_val.jet2_btagTight.values)

    
    mass_bins = np.quantile(X_train[Y_train==0].dijet_mass.values, np.linspace(0, 1, 15))
    mass_bins[0], mass_bins[-1] = 50., 300.
    if args.saveResults:
        np.save(outFolder+"/mass_bins.npy", mass_bins)
    if hp["bin_center_bool"]:
        bin_centers = [(mass_bins[i] + mass_bins[i+1]) / 2 for i in range(len(mass_bins) - 1)]


        bin_indices = np.digitize(X_train['dijet_mass'].values, mass_bins) - 1
        X_train['bin_center'] = np.where(
            (bin_indices >= 0) & (bin_indices < len(bin_centers)),  # Ensure valid indices
            np.array(bin_centers)[bin_indices],
            np.nan  # Assign NaN for out-of-range dijet_mass
        )
        bin_indices = np.digitize(X_val['dijet_mass'].values, mass_bins) - 1
        X_val['bin_center'] = np.where(
            (bin_indices >= 0) & (bin_indices < len(bin_centers)),  # Ensure valid indices
            np.array(bin_centers)[bin_indices],
            np.nan  # Assign NaN for out-of-range dijet_mass
        )
    print("Features used")
    print(featuresForTraining, flush=True)
    X_train = scale(X_train,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl", fit=True, scaler='robust')
    X_val  = scale(X_val, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl", fit=False, scaler='robust')

    size = hp['size']
    X_train, Y_train, W_train, rW_train, genMass_train, dijetMass_train, jet2_btagTight_train = X_train[:size], Y_train[:size], W_train[:size], rW_train[:size], genMass_train[:size], dijetMass_train[:size], jet2_btagTight_train[:size]
    X_val, Y_val, W_val, rW_val, genMass_val, dijetMass_val, jet2_btagTight_val = X_val[:size], Y_val[:size], W_val[:size], rW_val[:size], genMass_val[:size], dijetMass_val[:size], jet2_btagTight_val[:size]
    print(len(X_train), " events in train dataset")
    print(len(X_val), " events in val dataset")

    world_size = torch.cuda.device_count()
    if args.saveResults:
        summaryTxtPath = outFolder + "/model/training.txt"
    else:
        summaryTxtPath = "/scratch/sum_%s.txt"%args.modelName
    with open(summaryTxtPath, "w") as file:
        for key, value in hp.items():
            file.write(f"{key} : {value}\n")
        file.write(f"Mass Bins: {' '.join(f'{x:.1f}' for x in mass_bins)}\n")
        file.write("Lenght of X_train : %d\n"%len(X_train))
    
    print("Number of GPUs available is %d"%(world_size))
    mp.spawn(main, args=(world_size, hp, X_train, Y_train, W_train, rW_train, genMass_train, dijetMass_train, jet2_btagTight_train, X_val, Y_val, W_val, rW_val, genMass_val, dijetMass_val, jet2_btagTight_val, inFolder, outFolder, inputSubFolder, featuresForTraining, mass_bins, args.saveResults, args.modelName), nprocs=world_size)

    end_time = time.time()
    execution_time = end_time - start_time
    if args.saveResults:
        with open(summaryTxtPath, "a+") as file:
            file.write(f"Execution time: {execution_time} seconds\n")
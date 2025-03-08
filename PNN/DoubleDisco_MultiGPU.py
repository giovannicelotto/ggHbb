# Generic import
import numpy as np
import sys
import time
from datetime import datetime
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")

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
        val_classifier_loss_history     = []
        val_loss_history                = []
        val_dcor_loss_history           = []    

        batch_start_time = time.time()
        for epoch in range(hp["epochs"]):
            self.train_data.sampler.set_epoch(epoch)
            self.nn1.train()
            self.nn2.train()
            
            # Float for keeping track of Loss
            total_trainloss = 0.0
            total_trainclassifier_loss = 0.0
            total_traindcor_loss = 0.0

            
            # Training phase
            for batch in self.train_data:
                
                X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch = batch


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
                for low, high in zip(self.mass_bins[:-1], self.mass_bins[1:]):
                    # Mask for the current mass bin and bkg events only
                    bin_mask = (dijetMass_batch >= low) & (dijetMass_batch < high)  & (mask_batch)
                    assert bin_mask.size(0) == predictions1.size(0), f"Mismatch: bin_mask size {bin_mask.size(0)} vs predictions1 size {predictions1.size(0)}"
                    
                    #if (bin_mask.sum().item()==0):
                    #    print("No elements in this bin!")
                    #    continue
                    bin_predictions1 = predictions1[bin_mask]
                    bin_predictions2 = predictions2[bin_mask]
                    bin_weights = W_batch[bin_mask]

                    # Skip if there are no examples in the bin
                    if bin_predictions1.numel() == 0:
                        continue

                    # Compute dCorr for the current mass bin
                    dCorr_bin = distance_corr(bin_predictions1, bin_predictions2, bin_weights)

                    dCorr_total += dCorr_bin 

                # Combined loss
                loss = classifier_loss1 +classifier_loss2 + hp["lambda_dcor"] * dCorr_total/(len(self.mass_bins) -1) 
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

                with torch.no_grad():
                    for batch in self.val_data:
                        X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch = batch
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
                        for low, high in zip(self.mass_bins[:-1], self.mass_bins[1:]):
                            # Mask for the current mass bin
                            bin_mask = (dijetMass_batch >= low) & (dijetMass_batch < high) & (mask_batch)

                            # Apply bin-specific mask
                            bin_predictions1 = predictions1[bin_mask]
                            bin_predictions2 = predictions2[bin_mask]
                            bin_weights = W_batch[bin_mask]

                            # Compute dCorr for the current mass bin
                            dCorr_bin = distance_corr(bin_predictions1, bin_predictions2, bin_weights)
                            dCorr_total += dCorr_bin
                        # Combined loss
                        loss = classifier_loss1 + classifier_loss2 + hp["lambda_dcor"] * dCorr_total/(len(self.mass_bins) -1) 
                        total_val_loss += loss.item()
                        total_val_classifier_loss += classifier_loss1.item() + classifier_loss2.item()
                        total_val_dcor_loss += dCorr_total.item()/(len(self.mass_bins) -1)


            # Calculate average losses (average over batches)
            #print("len(self.train_data)", len(self.train_data))
            #print("len(self.val_data)", len(self.val_data))
            avg_trainloss = total_trainloss / len(self.train_data)
            avg_train_classifier_loss = total_trainclassifier_loss / len(self.train_data)
            avg_traindcor_loss = total_traindcor_loss / len(self.train_data)

            avg_val_loss = total_val_loss / len(self.val_data)
            avg_val_classifier_loss = total_val_classifier_loss / len(self.val_data)
            avg_val_dcor_loss = total_val_dcor_loss / len(self.val_data)
            
            train_loss_history.append(avg_trainloss)
            train_classifier_loss_history.append(avg_train_classifier_loss)
            train_dcor_loss_history.append(avg_traindcor_loss)
            val_loss_history.append(avg_val_loss)
            val_classifier_loss_history.append(avg_val_classifier_loss)
            val_dcor_loss_history.append(avg_val_dcor_loss)
            # Print losses
            if self.rank==0:
                print(f"Epoch [{epoch+1}/{hp['epochs']}], "
                  f"Train Loss: {avg_trainloss:.4f}, Classifier Loss: {avg_train_classifier_loss:.4f}, dCor Loss: {avg_traindcor_loss:.8f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Classifier Loss: {avg_val_classifier_loss:.4f}, Val dCor Loss: {avg_val_dcor_loss:.8f}",
                  flush=(epoch % 20 == 0))
                

            # CheckPoints
            if (self.rank==0) & (save) & (epoch%50==0):
                torch.save(self.nn1.state_dict(), outFolder + "/model/nn1_e%d.pth"%epoch)
                torch.save(self.nn2.state_dict(), outFolder + "/model/nn2_e%d.pth"%epoch)
            
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
         Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain, dijetMassTrain,
         Xval, Yval, Wval, rWval, genMassVal, dijetMassVal,
         inFolder, outFolder, inputSubFolder, featuresForTraining, mass_bins, save, modelName):
    

    
    # Assign each process to a GPU
    ddp_setup(rank, world_size)

    # Convert train and test into tensors in the corresponding GPUs
    # Change use rWtrain and rWval
    XtrainTensor = torch.tensor(Xtrain[featuresForTraining].values, dtype=torch.float32, device=rank)
    YtrainTensor = torch.tensor(Ytrain, dtype=torch.float, device=rank).unsqueeze(1)
    WtrainTensor = torch.tensor(rWtrain, dtype=torch.float32, device=rank).unsqueeze(1)
    dijetMassTrain_tensor = torch.tensor(dijetMassTrain, dtype=torch.float32, device=rank).unsqueeze(1)

    Xval_tensor = torch.tensor(Xval[featuresForTraining].values, dtype=torch.float32, device=rank)
    Yval_tensor = torch.tensor(Yval, dtype=torch.float, device=rank).unsqueeze(1)
    Wval_tensor = torch.tensor(rWval, dtype=torch.float32, device=rank).unsqueeze(1)
    dijetMassVal_tensor = torch.tensor(dijetMassVal, dtype=torch.float32, device=rank).unsqueeze(1)

    train_masks = (YtrainTensor < 0.5).to(rank)
    val_masks = (Yval_tensor < 0.5).to(rank)

    # Create TorchDataset
    traindataset = TensorDataset(
    XtrainTensor.to(rank),
    YtrainTensor.to(rank),
    WtrainTensor.to(rank),
    dijetMassTrain_tensor.to(rank),
    train_masks.to(rank)
    )

    val_dataset = TensorDataset(
        Xval_tensor.to(rank),
        Yval_tensor.to(rank),
        Wval_tensor.to(rank),
        dijetMassVal_tensor.to(rank),
        val_masks.to(rank)
    )


    hp["batch_size"] = hp["batch_size"] if hp["batch_size"]*world_size<len(Xtrain) else len(Xtrain)//world_size
    hp["val_batch_size"] = hp["batch_size"] if hp["batch_size"]*world_size<len(Xval) else len(Xval)//world_size
    print("Train Batch size", hp["batch_size"])
    print("Val Batch size", hp["val_batch_size"])
    # DataLoader(drop_last=True, sampler=DS(drop_last=False)) rules the total events e.g. size=850 and bs =100 8 GPUs. each GPU will see 1 batch with 100 events and 50 events epoch by eopch are not used.
    # DataLoader(drop_last=False, sampler=DS(drop_last=True)) . example before. each gpu will have 2 batches one with 100 and one with 6 8*100+8*6 = 848 and the additional 2 events are dropped
    # Given the dependence of disco wrt number of samples set all of them to True (equivalent to 1st=True and 2nd=False)
    # Check that samples not used change time by time. Epoch by epoch the events not used change! Because of shuffle=True inside DistributedSampler (default value)
    # Shuffle of datalaoder is mutually exclusive with sampler option. Inside DistributedSampler shuffle is True by default
    traindataloader = DataLoader(traindataset, batch_size=hp["batch_size"], shuffle=False, drop_last=True, pin_memory=False, sampler=DistributedSampler(traindataset, num_replicas=world_size, rank=rank, drop_last=True))
    val_dataloader = DataLoader(val_dataset, batch_size=hp["val_batch_size"], shuffle=False, drop_last=True, pin_memory=False, sampler=DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, drop_last=True))

    nn1 = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=hp["nNodes"])
    nn2 = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=hp["nNodes"])
    nn1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(nn1)
    nn2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(nn2)
    nn1.to(rank)
    nn2.to(rank)

    criterion = nn.BCELoss(reduction='none')
    optimizer1 = optim.Adam(nn1.parameters(), lr=hp["learning_rate"])
    optimizer2 = optim.Adam(nn2.parameters(), lr=hp["learning_rate"])
    
    # Define a trainer class
    trainer = Trainer(nn1=nn1, nn2=nn2, train_data=traindataloader, val_data=val_dataloader,
                      optimizer1=optimizer1, optimizer2=optimizer2, gpu_id=rank, criterion=criterion,
                      mass_bins=mass_bins)
    trainer.train(hp, outFolder=outFolder, save=save, modelName=modelName)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Script.")
    parser.add_argument("-l", "--lambda_dcor", type=str, help="lambda for penalty term", default=905)
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=5000)
    parser.add_argument("-s", "--size", type=int, help="Number of events to crop training dataset", default=3000000)
    parser.add_argument("-bs", "--batch_size", type=int, help="Number of eventsper batch size", default=32768)
    parser.add_argument("-n", "--nodes",type=lambda s: [int(item) for item in s.split(',')],  # Convert comma-separated string to list of ints
                        help="List of nodes per layer (e.g., 128,64,32 for a 3-layer NN)",default=None
)
    parser.add_argument("-sampling", "--sampling", type=int, help="input data from sampling signal method (True) or equal size of sample (false)", default=False)
    parser.add_argument("-bin_center", "--bin_center_bool", type=int, help="Use bin center as feature", default=True)
    parser.add_argument("-mH", "--massHypo", type=int, help="Use closest dijet mass discrete value as feature", default=True)
    parser.add_argument("-save", "--saveResults", type=int, help="saveResults (False in case of HP tuning)", default=True)
    parser.add_argument("-modelName", "--modelName", type=str, help="modelName (Only for HP tuning)", default='None')
    parser.add_argument("-lr", "--learningRate", type=str, help="learningRate", default=None)

    args = parser.parse_args()
    
    hp=getParams()
    hp["lambda_dcor"] = float(args.lambda_dcor )
    hp["epochs"] = args.epochs 
    hp["batch_size"] = int(args.batch_size )
    hp["size"]=args.size
    hp["sampling"] = args.sampling
    hp["bin_center_bool"] = args.bin_center_bool
    hp["massHypo"] = args.massHypo
    if args.nodes is not None:
        hp["nNodes"] = args.nodes
        print(hp["nNodes"])
    if args.learningRate is not None:
        hp["learning_rate"] = float(args.learningRate)
    if args.modelName != "None":
        print("ModelName %s"%args.modelName)
    print("ModelName %s"%args.modelName)


    print("Params changed")
    print_params(hp)

    start_time = time.time()
    current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'

    inFolder, outFolder = getInfolderOutfolder(name = "%s_%s"%(current_date, str(hp["lambda_dcor"]).replace('.', 'p')), suffixResults="DoubleDisco", createFolder=True if args.saveResults else False)
    inputSubFolder = 'data' if not hp["sampling"] else 'data_sampling'
    # Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
    if not args.saveResults:
        outFolder = "/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/commonFolder"
    featuresForTraining, columnsToRead = getFeatures(outFolder=outFolder, massHypo=False, bin_center=hp["bin_center_bool"], simple=True)

    Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval, Wtest, rWtrain, rWval, genMassTrain, genMassVal, genMassTest = loadXYWrWSaved(inFolder=inFolder+"/%s"%inputSubFolder)
    dijetMassTrain = np.array(Xtrain.dijet_mass.values)
    dijetMassVal = np.array(Xval.dijet_mass.values)
    print(len(Xtrain), " events in train dataset")
    print(len(Xval), " events in val dataset")
    mass_bins = np.quantile(Xtrain[Ytrain==0].dijet_mass.values, np.linspace(0, 1, 15))
    mass_bins[0], mass_bins[-1] = 40., 300.
    if args.saveResults:
        np.save(outFolder+"/mass_bins.npy", mass_bins)
    if hp["bin_center_bool"]:
        bin_centers = [(mass_bins[i] + mass_bins[i+1]) / 2 for i in range(len(mass_bins) - 1)]


        bin_indices = np.digitize(Xtrain['dijet_mass'].values, mass_bins) - 1
        Xtrain['bin_center'] = np.where(
            (bin_indices >= 0) & (bin_indices < len(bin_centers)),  # Ensure valid indices
            np.array(bin_centers)[bin_indices],
            np.nan  # Assign NaN for out-of-range dijet_mass
        )
        bin_indices = np.digitize(Xval['dijet_mass'].values, mass_bins) - 1
        Xval['bin_center'] = np.where(
            (bin_indices >= 0) & (bin_indices < len(bin_centers)),  # Ensure valid indices
            np.array(bin_centers)[bin_indices],
            np.nan  # Assign NaN for out-of-range dijet_mass
        )

    Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl", fit=True)
    Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl", fit=False)

    size = hp['size']
    Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain, dijetMassTrain = Xtrain[:size], Ytrain[:size], Wtrain[:size], rWtrain[:size], genMassTrain[:size], dijetMassTrain[:size]
    Xval, Yval, Wval, rWval, genMassVal, dijetMassVal = Xval[:size], Yval[:size], Wval[:size], rWval[:size], genMassVal[:size], dijetMassVal[:size]

    world_size = torch.cuda.device_count()
    if args.saveResults:
        summaryTxtPath = outFolder + "/model/training.txt"
    else:
        summaryTxtPath = "/scratch/sum_%s.txt"%args.modelName
    with open(summaryTxtPath, "w") as file:
        for key, value in hp.items():
            file.write(f"{key} : {value}\n")
        file.write(f"Mass Bins: {' '.join(f'{x:.1f}' for x in mass_bins)}\n")
        file.write("Lenght of Xtrain : %d\n"%len(Xtrain))
    
    print("Number of GPUs available is %d"%(world_size))
    mp.spawn(main, args=(world_size, hp, Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain, dijetMassTrain, Xval, Yval, Wval, rWval, genMassVal, dijetMassVal, inFolder, outFolder, inputSubFolder, featuresForTraining, mass_bins, args.saveResults, args.modelName), nprocs=world_size)

    end_time = time.time()
    execution_time = end_time - start_time
    if args.saveResults:
        with open(summaryTxtPath, "a+") as file:
            file.write(f"Execution time: {execution_time} seconds\n")
# %%
import numpy as np
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")

# PNN helpers
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.loadSaved import loadXYrWSaved
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale
from helpers.dcorLoss import *

# Torch
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    def __init__(
        self,
        model1: torch.nn.Module,  # First model (nn1)
        model2: torch.nn.Module,  # Second model (nn2)
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer1: torch.optim.Optimizer,  # Optimizer for nn1
        optimizer2: torch.optim.Optimizer,  # Optimizer for nn2
        gpu_id: int,
        #save_every: int,
        criterion: torch.nn.Module,  # Loss function
        hp: dict,  # Hyperparameters (including lambda_dcor, mass_bins, etc.)
    ) -> None:
        self.gpu_id = gpu_id
        self.model1 = model1.to(gpu_id)
        self.model2 = model2.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        #self.save_every = save_every
        self.criterion = criterion
        self.hp = hp
        self.model1 = DDP(self.model1, device_ids=[gpu_id])
        self.model2 = DDP(self.model2, device_ids=[gpu_id])

    def _run_batch(self, X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch, train=True):
        if train:
            # Reset the gradients of all optimized torch.Tensor (see https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()


        predictions1 = self.model1(X_batch)
        predictions2 = self.model2(X_batch)

        # Classifier Loss
        raw_loss1 = self.criterion(predictions1, Y_batch)
        raw_loss2 = self.criterion(predictions2, Y_batch)

        classifier_loss1 = (raw_loss1 * W_batch).mean()
        classifier_loss2 = (raw_loss2 * W_batch).mean()

        # dCorr computation
        W_batch = torch.ones([len(W_batch), 1], device=self.gpu_id) 
        dCorr_total = 0.
        
        for low, high in zip(self.hp["mass_bins"][:-1], self.hp["mass_bins"][1:]):
            # Mask for the current mass bin
            bin_mask = (dijetMass_batch >= low) & (dijetMass_batch < high) & mask_batch
            if bin_mask.sum().item() == 0:
                continue
            
            # Predictions and weights for the current mass bin
            bin_predictions1 = predictions1[bin_mask]
            bin_predictions2 = predictions2[bin_mask]
            bin_weights = W_batch[bin_mask]

            # Compute dCorr for the current bin
            dCorr_bin = distance_corr(bin_predictions1, bin_predictions2, bin_weights)
            dCorr_total += dCorr_bin

        # Compute total loss (combine classifier loss and dCorr)
        loss = classifier_loss1 + classifier_loss2 + self.hp["lambda_dcor"] * dCorr_total / (len(self.hp["mass_bins"]) - 1)

        if train:
            # Backpropagation if in training mode
            loss.backward()

            self.optimizer1.step()
            self.optimizer2.step()

        return loss.item(), classifier_loss1.item(), classifier_loss2.item(), dCorr_total.item()

    def _run_epoch(self, epoch):
        self.model1.train()
        self.model2.train()
        total_trainloss = 0.0
        total_trainclassifier_loss = 0.0
        total_traindcor_loss = 0.0
        # Training phase
        self.train_data.sampler.set_epoch(epoch)
        
        # Iterate through batches
        for X_batch, Y_batch, W_batch, advFeature_batch, dijetMass_batch, mask_batch in self.train_data:
            X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch = X_batch.to(self.gpu_id), Y_batch.to(self.gpu_id), W_batch.to(self.gpu_id), dijetMass_batch.to(self.gpu_id), mask_batch.to(self.gpu_id)
            
            # Run the batch and get the losses
            batch_loss, classifier_loss1, classifier_loss2, dCorr_loss = self._run_batch(
                X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch, train=True
            )
            
            # Accumulate loss
            total_train_loss += batch_loss
            total_classifier_loss += classifier_loss1 + classifier_loss2
            total_dcor_loss += dCorr_loss

            # Print epoch stats
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | "
                 f"Train Loss: {total_train_loss:.4f} | "
                 f"Classifier Loss: {total_classifier_loss:.4f} | "
                 f"dCorr Loss: {total_dcor_loss:.4f}")


# ______________________________________________________________________________
# ______________________________________________________________________________
# ______________________  VALIDATION PHASE  ____________________________________
# ______________________________________________________________________________
# ______________________________________________________________________________

    def validation_step(self, epoch):
        total_val_loss = 0.0
        total_val_classifier_loss = 0.0
        total_val_dcor_loss = 0.0

        # Set models to evaluation mode (turn off dropout, batch norm)
        self.model1.eval()
        self.model2.eval()

        with torch.no_grad():  # No gradients needed for validation
            for X_batch, Y_batch, W_batch, advFeature_batch, dijetMass_batch, mask_batch in self.val_data:
                X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch = X_batch.to(self.gpu_id), Y_batch.to(self.gpu_id), W_batch.to(self.gpu_id), dijetMass_batch.to(self.gpu_id), mask_batch.to(self.gpu_id)

                # Run the batch and get the losses (no backpropagation)
                batch_loss, classifier_loss1, classifier_loss2, dCorr_loss = self._run_batch(
                    X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch, train=False
                )

                # Accumulate validation loss
                total_val_loss += batch_loss
                total_val_classifier_loss += classifier_loss1 + classifier_loss2
                total_val_dcor_loss += dCorr_loss

        # Print validation stats
        print(f"[GPU{self.gpu_id}] Validation Loss: {total_val_loss:.4f} | "
              f"Classifier Loss: {total_val_classifier_loss:.4f} | "
              f"dCorr Loss: {total_val_dcor_loss:.4f}")
        

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # validation step after each epoch
            self.validation_step(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def main2(rank, world_size, device, current_date, hp):
    # First argument needs to be the rank automatically passed by the mp.spawn
    ddp_setup(rank, world_size)
    bin_center_bool = True
    inFolder, outFolder = getInfolderOutfolder(name = "%s_%s"%(current_date, str(hp["lambda_dcor"]).replace('.', 'p')), suffixResults="DoubleDisco")
    # Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
    featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=True, bin_center=bin_center_bool)
    
    Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, rWtrain, rWval, Wtest, genMassTrain, genMassVal, genMassTest = loadXYrWSaved(inFolder=inFolder+"/data")
    Wtrain, Wval = np.load(inFolder+"/data/Wtrain.npy"), np.load(inFolder+"/data/Wval.npy")
    dijetMassTrain = np.array(Xtrain.dijet_mass.values)
    dijetMassVal = np.array(Xval.dijet_mass.values)
    advFeatureTrain = np.load(inFolder+"/data/advFeatureTrain.npy")     
    advFeatureVal   = np.load(inFolder+"/data/advFeatureVal.npy")

    print(len(Xtrain), " events in train dataset")
    print(len(Xval), " events in val dataset")


    mass_bins = np.quantile(Xtrain[Ytrain==0].dijet_mass.values, np.linspace(0, 1, 15))
    mass_bins[0], mass_bins[-1] = 30., 300.
    np.save(outFolder+"/mass_bins.npy", mass_bins)
    if bin_center_bool:
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

    Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True)
    Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)

    # %%
    size = hp['size']
    Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain, advFeatureTrain, dijetMassTrain = Xtrain[:size], Ytrain[:size], Wtrain[:size], rWtrain[:size], genMassTrain[:size], advFeatureTrain[:size], dijetMassTrain[:size]
    Xval, Yval, Wval, rWval, genMassVal, advFeatureVal, dijetMassVal = Xval[:size], Yval[:size], Wval[:size], rWval[:size], genMassVal[:size], advFeatureVal[:size], dijetMassVal[:size]

    # %%
    # Comment if want to use flat in mjj
    rWtrain, rWval = Wtrain.copy(), Wval.copy()
    # %%

    XtrainTensor = torch.tensor(Xtrain[featuresForTraining].values, dtype=torch.float32, device=device)
    YtrainTensor = torch.tensor(Ytrain, dtype=torch.float, device=device).unsqueeze(1)
    rWtrainTensor = torch.tensor(rWtrain, dtype=torch.float32, device=device).unsqueeze(1)
    advFeatureTrain_tensor = torch.tensor(advFeatureTrain, dtype=torch.float32, device=device).unsqueeze(1)
    dijetMassTrain_tensor = torch.tensor(dijetMassTrain, dtype=torch.float32, device=device).unsqueeze(1)


    Xval_tensor = torch.tensor(Xval[featuresForTraining].values, dtype=torch.float32, device=device)
    Yval_tensor = torch.tensor(Yval, dtype=torch.float, device=device).unsqueeze(1)
    rWval_tensor = torch.tensor(rWval, dtype=torch.float32, device=device).unsqueeze(1)
    advFeatureVal_tensor = torch.tensor(advFeatureVal, dtype=torch.float32, device=device).unsqueeze(1)
    dijetMassVal_tensor = torch.tensor(dijetMassVal, dtype=torch.float32, device=device).unsqueeze(1)

    train_masks = (YtrainTensor < 0.5).to(device)
    val_masks = (Yval_tensor < 0.5).to(device)


    traindataset = TensorDataset(
    XtrainTensor.to(device),
    YtrainTensor.to(device),
    rWtrainTensor.to(device),
    advFeatureTrain_tensor.to(device),
    dijetMassTrain_tensor.to(device),
    train_masks.to(device)
    )
    val_dataset = TensorDataset(
        Xval_tensor.to(device),
        Yval_tensor.to(device),
        rWval_tensor.to(device),
        advFeatureVal_tensor.to(device),
        dijetMassVal_tensor.to(device),
        val_masks.to(device)
    )

    # Drop last to drop the last (if incomplete size) batch
    hp["batch_size"] = hp["batch_size"] if hp["batch_size"]<size else size
    #hp['bimod'] = 0
    traindataloader = DataLoader(traindataset, batch_size=hp["batch_size"], shuffle=True, drop_last=True, sampler=DistributedSampler(traindataset))
    val_dataloader = DataLoader(val_dataset, batch_size=hp["batch_size"], shuffle=False, drop_last=True, sampler=DistributedSampler(val_dataset))
    torch.distributed.barrier()
    # %%
    # Model, loss, optimizer
    nn1 = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=hp["nNodes"])
    nn2 = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=hp["nNodes"])
    # Move models to the correct GPU (using `rank` to select GPU)
    nn1.to(rank)
    nn2.to(rank)

    # Wrap the models with DDP
    nn1 = DDP(nn1, device_ids=[rank])
    nn2 = DDP(nn2, device_ids=[rank])
    epochs = hp["epochs"]
    criterion = nn.BCELoss(reduction='none')
    optimizer1 = optim.Adam(nn1.parameters(), lr=hp["learning_rate"])
    optimizer2 = optim.Adam(nn2.parameters(), lr=hp["learning_rate"])

    early_stopping_patience = hp["patienceES"]
    best_val_loss = float('inf')
    patience_counter = 0
    print("Train start")




    # Need now of
    # model: nn1 and nn2
    # dataloader: traindataloader, val_dataloader
    # optimizer: optimizer1, 
    # rank: rank
    # criterion
    
    trainer = Trainer(nn1, nn2, traindataloader, val_dataloader, optimizer1, optimizer2, rank, criterion, hp)
    trainer.train(hp["epochs"])
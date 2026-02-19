# %%
from GNN_class import GNN,GNN_3j1m,GNN_3j1m_hetero,GNN_3j1m_hetero_hetero, JetGraphDataset, myGAT
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys, argparse, os
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.doPlots import runPlotsTorch, doPlotLoss_Torch, plot_lossTorch
# %%
#model.load_state_dict(torch.load("/t3home/gcelotto/ggHbb/GNN/models/weights/gnn_v%d.pt"%version, map_location="cpu"))
parser = argparse.ArgumentParser(description="Script.")
#### Define arguments
parser.add_argument("-btagWP", "--btagWP", type=str, help="btagWP", default="M")
parser.add_argument("-v", "--modelVersion", type=int, help="modelVersion", default=4)
parser.add_argument("-l", "--lambda_disco", type=int, help="lambda_disco", default=10)
parser.add_argument("-c", "--coderDimension", type=int, help="coderDimension", default=8)


if 'ipykernel' in sys.modules:
    args = parser.parse_args(args=[])  # Avoids errors when running interactively
else:
    args = parser.parse_args()
model = myGAT(args.coderDimension)
folder = f"/work/gcelotto/GNN/model_{args.lambda_disco}_v{args.modelVersion}"
model.load_state_dict(torch.load(folder+"/GNN_weights/gnn.pt", map_location="cpu"))
#model.load_state_dict(torch.load(folder+"/GNN_weights/best_model_e92.pt", map_location="cpu"))
model.eval()


# %%
N = 1024*1024*8
dataset_train = torch.load("/t3home/gcelotto/ggHbb/GNN/graphs_train_hetero_hetero.pt")[:N]
dataset_val = torch.load("/t3home/gcelotto/ggHbb/GNN/graphs_val_hetero_hetero.pt")[:N]
Ytrain = np.load(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Ytrain_{args.btagWP}.npy")[:N]
genMassTrain = np.load(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/genMassTrain_{args.btagWP}.npy")[:N]
rWtrain = np.load(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/rWtrain_{args.btagWP}.npy")[:N]
rWval = np.load(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/rWval_{args.btagWP}.npy")[:N]
Xtrain = pd.read_parquet(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Xtrain_{args.btagWP}.parquet").iloc[:N,:]
Xval = pd.read_parquet(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Xval_{args.btagWP}.parquet").iloc[:N,:]
Wtrain = np.load(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Wtrain_{args.btagWP}.npy")[:N]
Wval = np.load(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Wval_{args.btagWP}.npy")[:N]
genMassVal = np.load(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/genMassVal_{args.btagWP}.npy")[:N]
Yval = np.load(f"/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Yval_{args.btagWP}.npy")[:N]
print("Loaded datasets with %d training and %d validation graphs" % (len(dataset_train), len(dataset_val)))
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"



model.eval() 
loader_train = DataLoader(dataset_train, batch_size=1024*8, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
loader_val = DataLoader(dataset_val, batch_size=1024*8, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
pred_train = []
pred_val = []


print("Preparing predictions")
with torch.inference_mode():  # disables gradient computation
    for idx, batch in enumerate(loader_train):
        print(
                f"\rProcessing train batch {idx+1} / {len(loader_train):d}   ",
                end="",
                flush=True,
            )


        batch = batch.to(device)

        out = model(batch)  # shape: [num_graphs_in_batch, num_classes]
        #preds = torch.softmax(out, dim=1)  # if you want probabilities
        pred_train.append(out)       # move to CPU
    for idx, batch in enumerate(loader_val):
        print(
                f"\rProcessing train batch {idx+1} / {len(loader_val):d}   ",
                end="",
                flush=True,
            )

        batch = batch.to(device)

        out = model(batch)  # shape: [num_graphs_in_batch, num_classes]
        #preds = torch.softmax(out, dim=1)  # if you want probabilities
        pred_val.append(out)       # move to CPU

# Concatenate all batches
pred_train = torch.cat(pred_train, dim=0) 
pred_val = torch.cat(pred_val, dim=0) 
# %%
fig, ax = plt.subplots(1,1)
bins = np.linspace(0, 1, 51)
ax.hist(torch.sigmoid(pred_train[Ytrain==0].cpu()), bins=bins, alpha=0.3, color='blue', label="bkg train", density=True)
ax.hist(torch.sigmoid(pred_train[genMassTrain==125].cpu()), bins=bins, alpha=0.3, color='red', label="signal train", weights=Wtrain[genMassTrain==125], density=True)
ax.hist(torch.sigmoid(pred_val[Yval==0].cpu()), bins=bins, histtype='step', color='blue', label="bkg val", density=True)
ax.hist(torch.sigmoid(pred_val[genMassVal==125].cpu()), bins=bins, histtype='step', color='red', label="signal val", weights=Wval[genMassVal==125], density=True)
ax.legend()
ax.set_xlabel("GNN output")
fig.savefig(folder+"/plots/pred_hist.png", bbox_inches='tight')
# %%
def sigmoid(z):
    return 1/(1 + np.exp(-z))
mistag_bkg = [10./100, 1./100, 0.5/100, 0.25/100, 0.1/100, 0]
for eff_low, eff_high in zip(mistag_bkg[:-1], mistag_bkg[1:]):

    eff_high =0
    thr_low = np.quantile(pred_val[Yval==0], 1 - eff_low)
    thr_high = np.quantile(pred_val[Yval==0], 1 - eff_high)
    signal_eff_train = (Wtrain[(pred_train.reshape(-1).numpy()> thr_low)& (pred_train.reshape(-1).numpy()< thr_high) & (genMassTrain==125) & (Xtrain.dijet_mass> 100) & (Xtrain.dijet_mass< 150)]).sum()/Wtrain[(genMassTrain==125) & (Xtrain.dijet_mass> 100) & (Xtrain.dijet_mass< 150)].sum()
    signal_eff_val = (Wval[(pred_val.reshape(-1).numpy()> thr_low) & (pred_val.reshape(-1).numpy()< thr_high) & (genMassVal==125) & (Xval.dijet_mass> 100) & (Xval.dijet_mass< 150)]).sum()/Wval[(genMassVal==125) & (Xval.dijet_mass> 100) & (Xval.dijet_mass< 150)].sum()
    print("Signal eff %.2f %% for background eff %.2f %% | thr = %.3f" % (signal_eff_val*100, eff_low*100, sigmoid(thr_low)))
    


    fig, ax = plt.subplots(1,1)
    bins = np.linspace(50, 200, 51)
    ax.hist(Xtrain.dijet_mass[(Ytrain==0) & (pred_train.reshape(-1).numpy()> thr_low)& (pred_train.reshape(-1).numpy()< thr_high)], bins=bins, alpha=0.3, color='blue', label="bkg train", density=True)
    ax.hist(Xtrain.dijet_mass[(genMassTrain==125) & (pred_train.reshape(-1).numpy()> thr_low)& (pred_train.reshape(-1).numpy()< thr_high)], bins=bins, alpha=0.3, color='red', label="signal train", weights=Wtrain[(genMassTrain==125) & (pred_train.reshape(-1).numpy()> thr_low)& (pred_train.reshape(-1).numpy()< thr_high)], density=True)
    ax.hist(Xval.dijet_mass[(Yval==0) & (pred_val.reshape(-1).numpy()> thr_low) & (pred_val.reshape(-1).numpy()< thr_high) ], bins=bins, histtype='step', color='blue', label="bkg val", density=True)
    ax.hist(Xval.dijet_mass[(genMassVal==125) & (pred_val.reshape(-1).numpy()> thr_low) & (pred_val.reshape(-1).numpy()< thr_high) ], bins=bins, histtype='step', color='red', label="signal val", weights=Wval[(genMassVal==125) & (pred_val.reshape(-1).numpy()> thr_low)& (pred_val.reshape(-1).numpy()< thr_high)], density=True)
    ax.text(0.95, 0.5, "Bkg $\epsilon$ : %.2f %%\nSig $\epsilon$ : %.2f %%\n" % (eff_low*100, signal_eff_val*100), transform=ax.transAxes, color='black', ha='right')
    ax.legend()
    ax.set_xlabel("Dijet mass [GeV]")
    ax.set_ylabel("Density")
    fig.savefig(folder+"/plots/mjj_thr%s_hist.png"%(str(eff_low).replace(".", "p")), bbox_inches='tight')
# %%
if os.path.exists(folder+"/losses_history/loss_total_train.npy"):
    train_loss_history = np.load(folder+"/losses_history/loss_total_train.npy")
    val_loss_history = np.load(folder+"/losses_history/loss_total_val.npy")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    ax.plot(train_loss_history, label='train (class + disco)')
    ax.plot(val_loss_history, label='val (class + disco)')

    ax.legend()
    fig.savefig(folder+"/plots/loss_total.png")


    loss_disco_history_train = np.load(folder+"/losses_history/loss_disco_train.npy")
    loss_disco_history_val = np.load(folder+"/losses_history/loss_disco_val.npy")

    train_classifier_loss_history = np.load(folder+"/losses_history/loss_classifier_train.npy")
    val_classifier_loss_history = np.load(folder+"/losses_history/loss_classifier_val.npy")

    fig, ax = plt.subplots(1,1)
    ax.plot(loss_disco_history_train, label='train')
    ax.plot(loss_disco_history_val, label='val')
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(np.min([loss_disco_history_train, loss_disco_history_val])*0.8, np.min([loss_disco_history_train, loss_disco_history_val])*20)
    fig.savefig(folder+"/plots/loss_disco.png")



    plot_lossTorch(train_loss_history, val_loss_history, 
                        train_classifier_loss_history, val_classifier_loss_history,
                        loss_disco_history_train, loss_disco_history_val,
                        train_closure_loss_history=None, val_closure_loss_history=None,
                        outFolder=folder+"/plots", gpu=False)

# %%

maskHiggsData_train = (genMassTrain==0) | (genMassTrain==125)
maskHiggsData_val = (genMassVal==0) | (genMassVal==125)
from sklearn.metrics import roc_curve, auc
fig, ax = plt.subplots()
def plot_roc_curve(y_true, y_scores, weights, label, ax, color=None, linestyle='solid'):
    fpr, tpr, _ = roc_curve(y_true, y_scores, sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label} (Weighted AUC = {roc_auc:.3f})", color=color, linestyle=linestyle)
    return roc_auc
plot_roc_curve(Ytrain[maskHiggsData_train], pred_train.reshape(-1).numpy()[maskHiggsData_train].ravel(), weights=Wtrain[maskHiggsData_train], label="Train", ax=ax)
plot_roc_curve(Yval[maskHiggsData_val], pred_val.reshape(-1).numpy()[maskHiggsData_val].ravel(),weights=Wval[maskHiggsData_val], label="Validation", ax=ax)
#plot_roc_curve((genMassTest==125).astype(int), YPredTest.ravel(), "Test", ax)
ax.plot([0, 1], [0, 1], 'k--')
ax.grid(True)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
fig.savefig(folder + "/plots/roc125_weighted.png", bbox_inches='tight')
print("Saved", folder + "/plots/roc125_weighted.png")
ax.set_xlim(0, 0.02)
ax.set_ylim(0,0.1)
fig.savefig(folder + "/plots/roc125_weighted_zoomed.png", bbox_inches='tight')
print("Saved", folder + "/plots/roc125_weighted_zoomed.png")



# %%

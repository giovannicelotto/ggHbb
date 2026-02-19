# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import yaml, sys
import torch
import dcor
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.loadSaved import loadXYWrWSaved
from helpers.scaleUnscale import scale, unscale
# %%
with open("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/signal_eff_vs_lambda_cfg.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    path = cfg["path"]
    models = cfg["models"]
    inFolder = cfg["inFolder"]

# %%
results = []
for model_lambda in models:
    for model in model_lambda:
        lambda_disco = int(model.split('_')[-1].split('p')[0]) 
        lambda_disco_tot = float(model.split('_')[-1].replace('p','.')) 
        print(path+f"/{model}/model/model.pth")
        torch_model = torch.load(path+f"/{model}/model/model.pth", map_location=torch.device('cpu'), weights_only=False)
        featuresForTraining = list(np.load(path+f"/{model}/model/featuresForTraining.npy"))
        for idx,f in enumerate(featuresForTraining):
            if f=="muon_pt":
                featuresForTraining[idx] = "jet1_muon_pt"
            elif f=="muon_dxySig":
                featuresForTraining[idx] = "jet1_muon_dxySig"
            elif f=="muon_pfRelIso03_all":
                featuresForTraining[idx] = "jet1_muon_pfRelIso03_all"
        Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genMassTrain, genMassVal = loadXYWrWSaved(inFolder=inFolder, isTest=False)
        Xval_s  = scale(Xval, featuresForTraining, scalerName= path+f"/{model}/model/myScaler.pkl" ,fit=False)
        Xtrain_s  = scale(Xtrain, featuresForTraining, scalerName= path+f"/{model}/model/myScaler.pkl" ,fit=False)
        Xval_tensor = torch.tensor(np.float32(Xval_s[featuresForTraining].values)).float()
        Xtrain_tensor = torch.tensor(np.float32(Xtrain_s[featuresForTraining].values)).float()
        with torch.no_grad():  # No need to track gradients for inference
            YPredTrain = torch_model(Xtrain_tensor).numpy()
            YPredVal = torch_model(Xval_tensor).numpy()

        # Find score at which bkg efficiency is (mistag) is 1%
        # save in a array (lambda, signal eff)
        # flatten scores
        scores = YPredVal.reshape(-1)
        weights = Wval
        # masks
        bkg_mask = (Yval == 0)
        sig_mask = (genMassVal == 125)


        # background scores and weights
        bkg_scores = scores[bkg_mask]
        bkg_weights = weights[bkg_mask]

        # sort background scores descending
        order = np.argsort(bkg_scores)[::-1]
        bkg_scores_sorted = bkg_scores[order]
        bkg_weights_sorted = bkg_weights[order]

        # cumulative background efficiency
        bkg_cum_eff = np.cumsum(bkg_weights_sorted) / np.sum(bkg_weights_sorted)

        # score threshold at 1% mistag
        idx = np.searchsorted(bkg_cum_eff, 0.005)
        score_cut = bkg_scores_sorted[idx]

        # signal efficiency at that cut
        sig_scores = scores[sig_mask]
        sig_weights = weights[sig_mask]
        dcor_value = dcor.distance_correlation(sig_scores[sig_scores >= score_cut], (Xval.dijet_mass.values[sig_mask])[sig_scores >= score_cut])

        sig_eff = np.sum(sig_weights[sig_scores >= score_cut]) / np.sum(sig_weights)

        # save (lambda, signal efficiency)
        results.append((lambda_disco, sig_eff, score_cut, dcor_value, lambda_disco_tot))
# %%
results=np.array(results)
x = results[:, 0]
y = results[:, 1]
cut = results[:, 2]
disCo = results[:, 3]
x_tot = results[:, -1]

unique_x = np.unique(x)

final = np.array([
    [
        xi,
        y[x == xi].mean(),
        y[x == xi].std(ddof=1) / np.sqrt((x == xi).sum()),
        cut[x == xi].mean(),
        cut[x == xi].std(ddof=1) / np.sqrt((x == xi).sum()),
        disCo[x == xi].mean(),
        disCo[x == xi].std(ddof=1) / np.sqrt((x == xi).sum()),
    ]
    for xi in unique_x
])

# %%
fig, ax = plt.subplots(1, 1)
ax.errorbar(final[:,0], final[:,1], final[:,2])
ax.errorbar(x_tot, results[:,1], marker='o', linestyle='none')
ax.set_xlabel("$\lambda$ disCo")
ax.set_ylabel("Signal eff. (@BkgEff 1%)")
# %%
fig, ax = plt.subplots(1, 1)
ax.errorbar(final[:,0], final[:,3], final[:,4])
ax.set_xlabel("$\lambda$ disCo")
ax.set_ylabel("Cut on NN score (@BkgEff 1%%)")
# %%
fig, ax = plt.subplots(1, 1)
ax.errorbar(final[:,0], final[:,5], final[:,6])
ax.set_xlabel("$\lambda$ disCo")
ax.set_ylabel("disCo on Events > cut (@BkgEff 1%%)")
# %%

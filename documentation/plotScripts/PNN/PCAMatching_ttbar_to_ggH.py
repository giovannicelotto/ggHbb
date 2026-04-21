# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
import numpy as np
hep.style.use("CMS")
import torch
import pandas as pd
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from scipy.stats import norm
# %%
folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"
modelName = "Jan21_3_50p0"
featuresForTraining = np.load(f"/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/{modelName}/model/featuresForTraining.npy")
# Open dataframes for ggH and ttbar
df = pd.read_parquet(folder + modelName + "/df_GluGluHToBBMINLO_Jan21_3_50p0.parquet")
df_tt = pd.read_parquet(folder + modelName + "/df_TTTo2L2Nu_Jan21_3_50p0.parquet", filters=[("is_ttbar_CR","==",1)])
# %%
# Scale datasets before making predictions consistently with training
from helpers.scaleUnscale import scale
df_ggH_scaled  = scale(df, featuresForTraining=featuresForTraining, scalerName= "/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/Jan21_3_50p0/model/myScaler.pkl" ,fit=False)
df_tt_scaled  = scale(df_tt, featuresForTraining=featuresForTraining, scalerName= "/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/Jan21_3_50p0/model/myScaler.pkl" ,fit=False)
nn = torch.load("/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/Jan21_3_50p0/model/model.pth", map_location=torch.device('cpu'))
nn.eval()
# Make predictions for ggH and ttbar
ggH_tensor = torch.tensor(np.float32(df_ggH_scaled[featuresForTraining].values)).float()
with torch.no_grad():  # No need to track gradients for inference
    ggH_predictions = nn(ggH_tensor).numpy()
tt_tensor = torch.tensor(np.float32(df_tt_scaled[featuresForTraining].values)).float()
with torch.no_grad():  # No need to track gradients for inference
    tt_predictions = nn(tt_tensor).numpy()

# %%
# Plot NN output before QM
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 1, 51)
ax.hist(ggH_predictions, bins=bins, density=True, histtype='step', label='ggH (SR)', color='blue')    
ax.hist(tt_predictions, bins=bins, density=True, histtype='step', label=r'$t\bar{t}$ ($t\bar{t}$ CR)', color='red')    
ax.set_xlabel("NN output")
ax.legend()







# %%
from sklearn.decomposition import PCA

X_tt = df_tt_scaled[featuresForTraining].values

pca = PCA()
Z_tt = pca.fit_transform(X_tt)

pc1_sigma = Z_tt[:, 0].std()
pc2_sigma = Z_tt[:, 1].std()
pc3_sigma = Z_tt[:, 2].std()

tt_predictions_var = {}
for i in range(-3,4,1):
    for j in range(-3,4,1):
        for k in range(-3,4,1):
            print(f"Varying PC1 by {i} sigma, PC2 by {j} sigma, PC3 by {k} sigma")
            Z_tt_var = Z_tt.copy()
            Z_tt_var[:, 0] += i*pc1_sigma
            Z_tt_var[:, 1] += j*pc2_sigma
            Z_tt_var[:, 2] += k*pc3_sigma

            X_tt_var = pca.inverse_transform(Z_tt_var)


            df_tt_pca_var = df_tt_scaled.copy()
            df_tt_pca_var[featuresForTraining] = X_tt_var


            tt_tensor_var = torch.tensor(
                np.float32(df_tt_pca_var[featuresForTraining].values)
            ).float()

            with torch.no_grad():
                tt_predictions_var[f'{i},{j},{k}'] = nn(tt_tensor_var).numpy()





# %%
fig, ax = plt.subplots(1, 1)

bins = np.array([0,0.7, 0.8,0.9,0.95, 1])
countsInLastBin = -1
PCA_variations = [9,-9,-9]
#bins = np.linspace(0, 1, 101)
ax.hist(tt_predictions, bins=bins, density=True, histtype='step', label='Nominal', linewidth=3, color='black')
for i in range(-3,4,1):
    for j in range(-3,4,1):
        for k in range(-3,4,1):
            if np.histogram(tt_predictions_var[f'{i},{j},{k}'], bins=bins)[0][-2] < np.histogram(tt_predictions, bins=bins)[0][-2]*13:
                continue
            currentCounts = ax.hist(tt_predictions_var[f'{i},{j},{k}'], bins=bins, density=False, histtype='step', label=f'{i},{j},{k}')[0][-1]
            if currentCounts > countsInLastBin:
                countsInLastBin = currentCounts
                PCA_variations = [i,j,k]
ax.set_yscale('log')
ax.set_xlabel("NN output")
ax.set_xlim(.8, 1)
ax.legend()


# %%

import joblib

joblib.dump(pca, "/t3home/gcelotto/ggHbb/tt_CR/PCA/pca_ttbar.pkl")
import yaml

variation_config = {
    "pc1_sigma": float(pc1_sigma),
    "pc2_sigma": float(pc2_sigma),
    "pc3_sigma": float(pc3_sigma),
    "shift": {
        "pc1":  PCA_variations[0],
        "pc2": PCA_variations[1],
        "pc3": PCA_variations[2]
    },
    #"features": list(featuresForTraining),
    "pca_file": "/t3home/gcelotto/ggHbb/tt_CR/PCA/pca_ttbar.pkl",
    "scaler_file": "myScaler.pkl"
}

# Save to YAML
with open("/t3home/gcelotto/ggHbb/tt_CR/PCA/pca_variation.yaml", "w") as f:
    yaml.dump(variation_config, f)






 
# %%

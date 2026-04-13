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
import sys
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts")
from plotFeatures import *
# %%
folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/"
modelName = "Jan21_3_50p0"
featuresForTraining = np.load(f"/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/{modelName}/model/featuresForTraining.npy")
# %%
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
ax.hist(ggH_predictions, bins=bins, density=True, histtype='step', label='ggH (SR)', color='red')    
ax.hist(tt_predictions, bins=bins, density=True, histtype='step', label=r'$t\bar{t}$ ($\ell\bar{\ell}$ CR)', color='blue')    
ax.set_xlabel("NN output")
ax.legend()





# %%






 














# %%
print("*********\nNaive Quantile Matching\n*********")
from sklearn.preprocessing import QuantileTransformer
# Fit quantile matching to ggH and ttbar to make uniform distribution
qt_nom = QuantileTransformer(
    output_distribution="uniform",
    random_state=0
)

qt_tt = QuantileTransformer(
    output_distribution="uniform",
    random_state=0
)

qt_nom.fit(df_ggH_scaled[featuresForTraining])
qt_tt.fit(df_tt_scaled[featuresForTraining])

# ttbar -> uniform (using ttbar CDF)
ttbar_uniform = qt_tt.transform(
    df_tt_scaled[featuresForTraining]
)

# uniform -> nominal (using nominal inverse CDF of ggH)
ttbar_mapped_to_nominal = qt_nom.inverse_transform(
    ttbar_uniform
)
# %%
# Create a dataframe with the morphed features, keeping the same index as the original ttbar dataframe
df_tt_morphed = pd.DataFrame(
    ttbar_mapped_to_nominal,
    columns=featuresForTraining,
    index=df_tt_scaled.index
)
    
# %%

featuresForTraining = np.load(f"/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/{modelName}/model/featuresForTraining.npy")













# %%
print("*********\nQuantile Matching + Copula Matching\n*********")
features = featuresForTraining

qt_tt = QuantileTransformer(
    n_quantiles=1000,
    output_distribution="uniform",
    random_state=0
)

qt_ggH = QuantileTransformer(
    n_quantiles=1000,
    output_distribution="uniform",
    random_state=0
)

qt_tt.fit(df_tt_scaled[features])


# %%
eps = 1e-6

U_tt = qt_tt.transform(df_tt_scaled[features])
U_tt = np.clip(U_tt, eps, 1 - eps)

Z_tt = norm.ppf(U_tt)   # now standard normal marginals

# %%

# whiten ttbar
cov_tt = np.cov(Z_tt[:len(Z_tt)//100,:], rowvar=False)
L_tt = np.linalg.cholesky(cov_tt)

Z_tt_whitened = np.linalg.solve(L_tt, Z_tt.T).T

# covariance of ggH in normal space
qt_ggH.fit(df_ggH_scaled[features])
U_ggH = qt_ggH.transform(df_ggH_scaled[features])
U_ggH = np.clip(U_ggH, eps, 1 - eps)
#plotNormalizedFeatures(data=[pd.DataFrame(U_ggH, columns=featuresForTraining), 
#                             pd.DataFrame(U_tt, columns=featuresForTraining)], outFile=None, legendLabels=['ggH Uniform', 'ttbar Uniform'], colors=['red', 'blue'], histtypes=None, alphas=None, figsize=None, autobins=True,
#                       weights=[df.flat_weight, df_tt.flat_weight], error=True)
# %%
Z_ggH = norm.ppf(U_ggH)
#plotNormalizedFeatures(data=[pd.DataFrame(Z_ggH, columns=featuresForTraining), 
#                             pd.DataFrame(Z_tt, columns=featuresForTraining)], outFile=None, legendLabels=['ggH Uniform', 'ttbar Uniform'], colors=['red', 'blue'], histtypes=None, alphas=None, figsize=None, autobins=True,
#                       weights=[df.flat_weight, df_tt.flat_weight], error=True)
cov_ggH = np.cov(Z_ggH[:len(Z_ggH)//100], rowvar=False)

# Cholesky decomposition
L_ggH = np.linalg.cholesky(cov_ggH)
# recolor with ggH covariance
Z_tt_corr = Z_tt_whitened @ L_ggH.T

# %%
U_tt_corr = norm.cdf(Z_tt_corr)

X_tt_morphed = qt_ggH.inverse_transform(U_tt_corr)

df_tt_morphed_copula = pd.DataFrame(
    X_tt_morphed,
    columns=features,
    index=df_tt_scaled.index
)
# %%
import pandas as pd
import matplotlib.pyplot as plt

# Suppose df is your dataframe with features
corr = pd.DataFrame(Z_ggH, columns=featuresForTraining).corr()  # computes correlation matrix

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)

# Set tick labels
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

plt.title("Feature Correlation Matrix ggH")
plt.show()




corr = pd.DataFrame(Z_tt, columns=featuresForTraining).corr()  # computes correlation matrix
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
# Set tick labels
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
plt.title("Feature Correlation Matrix tt")
plt.show()




corr = pd.DataFrame(Z_tt_whitened, columns=featuresForTraining).corr()  # computes correlation matrix
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
# Set tick labels
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
plt.title("Feature Correlation Matrix tt whitened")
plt.show()

# %%
#plotNormalizedFeatures(data=[df_ggH_scaled[featuresForTraining], df_tt_morphed_copula[featuresForTraining]], outFile=None, legendLabels=['ggH original', 'ttbar morphed'], colors=['blue', 'red'], histtypes=None, alphas=None, figsize=None, autobins=False,
#                       weights=[df.flat_weight, df_tt.flat_weight], error=True)
# %%
tt_tensor_morphed_copula = torch.tensor(np.float32(df_tt_morphed_copula[featuresForTraining].values)).float()
with torch.no_grad():  # No need to track gradients for inference
    tt_predictions_morphed_copula = nn(tt_tensor_morphed_copula).numpy()
tt_tensor_morphed_simple = torch.tensor(np.float32(df_tt_morphed[featuresForTraining].values)).float()
with torch.no_grad():  # No need to track gradients for inference
    tt_predictions_morphed_simple = nn(tt_tensor_morphed_simple).numpy()
# %%
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 1, 51)
ax.hist(ggH_predictions, bins=bins, density=True, histtype='step', label='ggH (SR)', color='red', linewidth=2)    
ax.hist(tt_predictions, bins=bins, density=True, histtype='step', label=r'$t\bar{t}$ ($\ell\bar{\ell}$ CR)', color='blue', linewidth=2)    
ax.hist(tt_predictions_morphed_simple, bins=bins, density=True, histtype='step', label=r'$t\bar{t}$  ($\ell\bar{\ell}$ CR after QM)', color='green', linewidth=2)    
ax.hist(tt_predictions_morphed_copula, bins=bins, density=True, histtype='step', label=r'$t\bar{t}$ QM+cov ($\ell\bar{\ell}$ CR)', color='purple', linewidth=2)    

ax.set_xlabel("NN output")
ax.legend()
# %%
import joblib
import numpy as np

# Save QuantileTransformers
#joblib.dump(qt_tt,   "/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/qt_tt.pkl")
#joblib.dump(qt_ggH,  "/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/qt_ggH.pkl")
## Save Cholesky matrices
#np.save("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/L_tt.npy",  L_tt)
#np.save("/t3home/gcelotto/ggHbb/documentation/plotScripts/PNN/quantile_matching/L_ggH.npy", L_ggH)
# %%

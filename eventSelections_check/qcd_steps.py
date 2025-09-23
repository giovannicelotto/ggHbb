# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functions import *
import mplhep as hep
hep.style.use("CMS")
import torch
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.scaleUnscale import scale
# %%
dfProcess = getDfProcesses_v2()[0]
featureNorm = "Total x-section"
# %%
# Definizione selezioni (comuni a QCD e Higgs)
selections = {
    featureNorm: lambda df: df,
    "Trigger": lambda df: df,
    "105 < mjj < 140": lambda df: df[(df.dijet_mass > 105) & (df.dijet_mass < 140)],
    "B-tag": lambda df: df[
        (df.dijet_mass > 105) & (df.dijet_mass < 140) &
        (df.jet1_btagDeepFlavB > 0.71) &
        (df.jet2_btagDeepFlavB > 0.71)
    ],
    "Kinematic Cuts": lambda df: df[
        (df.dijet_mass > 105) & (df.dijet_mass < 140) &
        (df.dijet_pt > 100) &
        (df.jet1_btagDeepFlavB > 0.71) &
        (df.jet2_btagDeepFlavB > 0.71) &
        (df.muon_pt > 9) &
        (abs(df.muon_dxySig) > 6)
    ],

    "NN cut": lambda df: df[
        (df.dijet_mass > 105) & (df.dijet_mass < 140) &
        (df.dijet_pt > 100) &
        (df.jet1_btagDeepFlavB > 0.71) &
        (df.jet2_btagDeepFlavB > 0.71) &
        (df.muon_pt > 9) &
        (abs(df.muon_dxySig) > 6) &
        (df.NN > 0.7)
    ]
}

# Funzione per calcolare cross-sections
def compute_xsections(MCList, selections):
    dfsMC, sumw = loadMultiParquet_v2(paths=MCList, nMCs=10, filters=None, returnNumEventsTotal=True)


    # Get NN predictions

    modelName = "Jul15_3_10p0"
    modelDir = "/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s/model"%(modelName)
    featuresForTraining = list(np.load("/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s/featuresForTraining.npy"%modelName))
    nn = torch.load(modelDir+"/model.pth", map_location=torch.device('cpu'))
    for idx, df in enumerate(dfsMC):
        df_scaled  = scale(df, featuresForTraining=featuresForTraining, scalerName= modelDir + "/myScaler.pkl" ,fit=False)
        nn.eval()
        data_tensor = torch.tensor(np.float32(df_scaled[featuresForTraining].values)).float()
    

        with torch.no_grad():  # No need to track gradients for inference
            data_predictions1 = nn(data_tensor).numpy()
        dfsMC[idx]['NN'] = data_predictions1
        
    

    


    results = {}
    for label, selection in selections.items():
        totalXSection = 0
        for idx, df in enumerate(dfsMC):
            df_sel = selection(df)
            if label==featureNorm:
                print("label is ", label )
                if 37 in MCList:
                    xsection_pass = 28.19
                else:
                    xsection_pass = 187300000.0 + 23590000.0
            else:
                xsection_pass = (df_sel.genWeight * df_sel.sf * df_sel.PU_SF).sum() / sumw[idx] * dfProcess.iloc[MCList].iloc[idx].xsection

            totalXSection += xsection_pass
        results[label] = totalXSection
    return results, dfsMC

# %%
# Calcolo per QCD e Higgs
xsec_qcd, dfs_QCD   = compute_xsections([23,24,25,26,27,28,29,30,31,32,33,34], selections)
xsec_higgs, dfs_Higgs = compute_xsections([37], selections)

# Metto in DataFrame
df_plot = pd.DataFrame({
    "QCD": xsec_qcd,
    "Higgs": xsec_higgs
})

# Normalizzo rispetto al "No selection"
df_plot_norm = df_plot.div(df_plot.loc[featureNorm])
# %%
# Plot
fig, ax = plt.subplots(1, 1)
df_plot_norm.plot(kind="bar", ax=ax)
plt.ylabel("Relative cross-section")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
ax.set_yscale('log')
hep.cms.label("Preliminary",data=False)
plt.show()
# %%

















# %%

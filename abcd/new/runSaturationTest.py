# %%
import pandas as pd
import sys
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
from plotDfs import plotDfs
from helpersABCD.abcd_maker_v2 import ABCD
from functions import getDfProcesses, cut, getDfProcesses_v2, cut_advanced
import numpy as np
import argparse
from helpersABCD.plot_v2 import pullsVsDisco, pullsVsDisco_pearson
import matplotlib.pyplot as plt
from helpersABCD.saturationStd import saturationTest
import pickle
# %%
parser = argparse.ArgumentParser(description="Script.")

modelName = "Apr01_1000p0"
dd = True
outFolder = "/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s"%modelName
df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/%s"%modelName
bins = np.load(outFolder+"/mass_bins.npy")
#midpoints = (bins[:-1] + bins[1:]) / 2
#bins = np.sort(np.concatenate([bins, midpoints]))
#bins=bins[1:]
#bins[-1]=300
#bins=bins[:-1]
#midpoints = (bins[:-1] + bins[1:]) / 2
#bins = np.sort(np.concatenate([bins, midpoints]))

# %%
dfs = []
dfProcessesMC, dfProcessesData = getDfProcesses_v2()
# %%
# Loading data
dfsMC = []
isMCList = [0,
            1, 
            2,3,
            #4,
            #5,6,7,8, 9,10,
            #11,12,13,
            #14,15,16,17,18,
            19,20,21, 22, 35,
            36
            ]
for idx, p in enumerate(dfProcessesMC.process):
    if idx not in isMCList:
        continue
    df = pd.read_parquet(df_folder+"/df_%s%s_%s.parquet"%("dd_" if dd else "", p, modelName))
    dfsMC.append(df)
#dcor_MC_values =  dcor_plot_Data(dfsMC, dfProcessesMC.process, isMCList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcorMC_%s_%s.png"%(modelName, detail), nEvents=-1)
# %%
dfsData = []
isDataList = [
        0,
        1,
        2,
        3,
        4,
        #5,
           # 6
            ]

lumis = []
for idx, p in enumerate(dfProcessesData.process):
    if idx not in isDataList:
        continue
    df = pd.read_parquet(df_folder+"/dataframes%s%s_%s.parquet"%("_dd_" if dd else "", p, modelName))
    dfsData.append(df)
    lumi = np.load(df_folder+"/lumi%s%s_%s.npy"%("_dd_", p, modelName))
    lumis.append(lumi)
lumi = np.sum(lumis)
for idx, df in enumerate(dfsMC):
    dfsMC[idx].weight =dfsMC[idx].weight*lumi



dfsMC = cut(dfsMC, 'muon_pt', 9, None)
dfsData = cut(dfsData, 'muon_pt', 9, None)

    # %%
x1 = 'PNN1' if dd else 'jet1_btagDeepFlavB'
x2 = 'PNN2' if dd else 'PNN'
xx = 'dijet_mass'

# %%
from scipy.stats import pearsonr
pearson_data_values = []
df = pd.concat(dfsData)

# %%

stds = saturationTest(bins,df, bootstrapEvents=[100,300,800], fractions=[0.25, 0.5, 0.75])
with open("/t3home/gcelotto/ggHbb/abcd/new/output/stds.pkl", "wb") as f:
    pickle.dump(stds, f)
# %%
with open("/t3home/gcelotto/ggHbb/abcd/new/output/stds.pkl", "rb") as f:
    stds = pickle.load(f)
# %%
bin_idx = np.array(stds['bin_idx'])
fraction = np.array(stds['fraction'])
nBoot = np.array(stds['nBoot'])
std = np.array(stds['std'])
fig, ax = plt.subplots(1, 1)
mask1 = (bin_idx==5) & (nBoot==100)
mask3 = (bin_idx==5) & (nBoot==300)
mask8 = (bin_idx==5) & (nBoot==800)
ax.plot(fraction[mask1], std[mask1], marker='o', label='100')
ax.plot(fraction[mask3], std[mask3], marker='o', label='300')
ax.plot(fraction[mask8], std[mask8], marker='o', label='800')
ax.set_xlabel("fraction")
ax.legend()
# %%

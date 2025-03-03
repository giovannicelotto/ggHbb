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
# %%
parser = argparse.ArgumentParser(description="Script.")
try:
    parser.add_argument("-m", "--modelName", type=str, help="e.g. Dec19_500p9", default=None)
    parser.add_argument("-dd", "--doubleDisco", type=bool, help="Single Disco (False) or double Disco (True). If false use jet1btag as variable", default=False)
    args = parser.parse_args()
    if args.modelName is not None:
        modelName = args.modelName
    dd = args.doubleDisco
except:
    print("Interactive mode")
    modelName = "Feb24_900p0"
    dd = True
outFolder = "/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s"%modelName
df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/%s"%modelName
bins = np.load(outFolder+"/mass_bins.npy")
#bins=np.linspace(40, 300, 6)
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
            2,3, 4,
            5,6,7,8, 9,10,
            11,12,13,
            14,15,16,17,18,
            19,20,21, 22, 35,
            36,
    # Spin0 200
    #41#
            ]
for idx, p in enumerate(dfProcessesMC.process):
    if idx not in isMCList:
        continue
    df = pd.read_parquet(df_folder+"/df_%s%s_%s.parquet"%("dd_" if dd else "", p, modelName))
    dfsMC.append(df)

# %%
dfsBackground = []
    
isMCListBackground = [
    #0,
    #1, 
    #2,3, 4,
    #5,6,7,8, 9,10,
    #11,12,13,
    #14,15,16,17,18,
    #19,20,21, 22, 35,
    23, 24, 25, 26,
    27,
    28, 29, 30, 31, 32, 33, #34,
    #36,
# Spin0 200
    #41
    
    ]

for idx, p in enumerate(dfProcessesMC.process):
    if idx not in isMCListBackground:
        print("Excl. ", p)
        continue
    print("Taken. ", p)
    df = pd.read_parquet(df_folder+"/df_%s%s_%s.parquet"%("dd_" if dd else "", p, modelName))
    dfsBackground.append(df)
# %%
# To make the MC independent keep onlt the first half. The second hald will be used for Background
dfsMC = [df.iloc[:len(df) // 2] for df in dfsMC]
for idx, (df, isMC) in enumerate(zip(dfsMC, isMCList)):
    if isMC == 41:
        dfsMC[idx].loc[:, "weight"] *= 10
    else:
        dfsMC[idx].loc[:, "weight"] *= 2

for idx, (df, isMC) in enumerate(zip(dfsBackground, isMCListBackground)):
    if isMC in isMCList:
        print(dfProcessesMC.process[isMC])
        dfsBackground[idx] = df.iloc[len(df) // 2:]
        if isMC == 41:
            dfsBackground[idx].loc[:, "weight"] *= 10
        else:
            dfsBackground[idx].loc[:, "weight"] *= 2

    else:
        pass
# %%
x1 = 'PNN2' if dd else 'jet1_btagDeepFlavB'
x2 = 'PNN1' if dd else 'PNN'
xx = 'dijet_mass'
# tight WP for btag 0.7100
t1 = 0.5
t2 = 0.5


# %%
from helpersABCD.dcorPlot_process_datataking import dcor_plot_Data, dpearson_plot_Data


#dfsData = cut_advanced(dfsData, 'jet1_nTightMuons', 'abs(jet1_nTightMuons) < 1.1')
#dfsMC = cut_advanced(dfsMC, 'jet1_nTightMuons', 'abs(jet1_nTightMuons) < 1.1')
#dfsData = cut_advanced(dfsData, 'dijet_eta', 'abs(dijet_eta) > 3.0')
#dfsMC = cut_advanced(dfsMC, 'dijet_eta', 'abs(dijet_eta) > 3.0')

#dfsData = cut(dfsData, 'ht', 200, None)
#dfsMC = cut(dfsMC, 'ht', 200, None)

detail = 'QCD'
# %%
#dcor_plot_Data(dfsBackground, dfProcessesMC.process, isMCListBackground, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcor%s.png"%detail)
fig = plotDfs(dfsData=dfsBackground, dfsMC=dfsMC, isMCList=isMCList, dfProcesses=dfProcessesMC, nbin=101, lumi=1, log=True, blindPar=(False, 105, 30))
dpearson_plot_Data([pd.concat(dfsBackground)], ['QCD'], isMCListBackground, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dpearsonR/pearson%s.png"%detail)
# %%
pulls_QCD_SR, err_pulls_QCD_SR = ABCD(dfsBackground, dfsMC,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=1, suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail), blindPar=(True, 100.5, 40))
# %%
from scipy.stats import pearsonr
def weighted_pearsonr(x, y, w):
    x_mean = np.sum(w * x) / np.sum(w)
    y_mean = np.sum(w * y) / np.sum(w)
    
    cov_xy = np.sum(w * (x - x_mean) * (y - y_mean))
    std_x = np.sqrt(np.sum(w * (x - x_mean) ** 2))
    std_y = np.sqrt(np.sum(w * (y - y_mean) ** 2))
    pearsonr = cov_xy / (std_x * std_y)
    print(pearsonr)
    return pearsonr
pearson_data_values = []
df = pd.concat(dfsBackground)
for b_low, b_high in zip(bins[:-1], bins[1:]):
    m = (df.dijet_mass > b_low) & (df.dijet_mass < b_high)
    x = np.array(df.PNN1[m], dtype=np.float64)
    y = np.array(df.PNN2[m], dtype=np.float64)
    w = np.array(df.weight[m], dtype=np.float64)
    pearson_coef = weighted_pearsonr(x, y, w)
    pearson_data_values.append(pearson_coef)
    print("     %.1f < mjj < %.1f : %.5f"%(b_low, b_high, pearson_coef))
pearson_data_values = np.array(pearson_data_values)
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
maskBlind = pulls_QCD_SR>0
condition1 = pearson_data_values[pulls_QCD_SR>0] > 0
condition2 = pulls_QCD_SR[pulls_QCD_SR>0]-1 > 0
yTrue = pulls_QCD_SR[pulls_QCD_SR>0]-1
ax.bar(((bins[:-1]+bins[1:])/2)[maskBlind], np.where(condition1, yTrue, -yTrue), width=1, label='Sign of pearson')
ax.bar(((bins[:-1]+bins[1:])/2)[maskBlind], np.where(condition2, yTrue*0.5, -yTrue*0.5), width=1, label='Sign of Pull')
ax.legend()
# %%

m = np.abs(pulls_QCD_SR)>0
popt, pcov = pullsVsDisco_pearson(pearson_data_values, pulls_QCD_SR, err_pulls_QCD_SR, mask =m, lumi=0, outName="/t3home/gcelotto/ggHbb/abcd/new/plots/pulls_vs_dcor/pulls_vs_dPearson_%s_%s.png"%(modelName, detail))
corrections = popt[1]*pearson_data_values + popt[0]
detailC = detail+'_corrected'
corrections = 1/corrections
# %%
pulls_QCD_SR = ABCD(dfsBackground, dfsMC,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=1, suffix='%s_%s_%s'%(modelName, "dd" if dd else "", detailC), blindPar=(False, 120.5, 20), corrections=corrections)

# %%
print("m : ", popt[1], " +- ", np.sqrt(pcov[1,1]))
print("q : ", popt[0], " +- ", np.sqrt(pcov[0,0]))
# %%

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
    modelName = "Feb05_900p0"
    dd = True
outFolder = "/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s"%modelName
df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/%s"%modelName
bins = np.load(outFolder+"/mass_bins.npy")
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
            36
            ]
for idx, p in enumerate(dfProcessesMC.process):
    if idx not in isMCList:
        continue
    df = pd.read_parquet(df_folder+"/df_%s%s_%s.parquet"%("dd_" if dd else "", p, modelName))
    dfsMC.append(df)

# %%
dfsData = []
isDataList = [
             0,
             1, 
            #2,
            3,
           # 4,
           # 5,
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
# %%
detail = ''
for f in dfProcessesData.process[isDataList].values:
    detail = detail + "_"+f[4:]

#dcor_plot_Data(dfsData, dfProcessesData.process, isDataList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcor%s.png"%detail)
fig = plotDfs(dfsData=dfsData, dfsMC=dfsMC, isMCList=isMCList, dfProcesses=dfProcessesMC, nbin=101, lumi=lumi, log=True, blindPar=(True, 105, 30))
#dpearson_plot_Data([pd.concat(dfsData)], ['Data'], isDataList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dpearsonR/pearson%s.png"%detail)
# %%
pulls_QCD_SR, err_pulls_QCD_SR = ABCD(dfsData, dfsMC,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=lumi, suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail), blindPar=(True, 100.5, 40))
# %%
from scipy.stats import pearsonr
pearson_data_values = []
df = pd.concat(dfsData)
for b_low, b_high in zip(bins[:-1], bins[1:]):
    m = (df.dijet_mass > b_low) & (df.dijet_mass < b_high)
    x = np.array(df.PNN1[m], dtype=np.float64)
    y = np.array(df.PNN2[m], dtype=np.float64)
    pearson_coef = pearsonr(x, y)[0]
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
pulls_QCD_SR = ABCD(dfsData, dfsMC,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=lumi, suffix='%s_%s_%s'%(modelName, "dd" if dd else "", detailC), blindPar=(True, 120.5, 20), corrections=corrections)

# %%
print("m : ", popt[1], " +- ", np.sqrt(pcov[1,1]))
print("q : ", popt[0], " +- ", np.sqrt(pcov[0,0]))
# %%

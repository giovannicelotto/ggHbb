# %%
import pandas as pd
import sys
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
from plotDfs import plotDfs
from helpersABCD.abcd_maker_v2 import ABCD
from functions import  cut, getDfProcesses_v2
import numpy as np
import argparse
from helpersABCD.plot_v2 import pullsVsDisco, pullsVsDisco_pearson
import matplotlib.pyplot as plt
from helpersABCD.plotNNscore import plotNNscore
from helpersABCD.runPearson import runPearson, makePlotsPearson
from helpersABCD.computeCorrections import computeCorrections
from scipy.stats import pearsonr
from helpersABCD.getStdBinABCD_bootstrap import *
from helpersABCD.getDataFrames import getDataFrames
import yaml
from helpersABCD.dcorPlot_process_datataking import dcor_plot_Data
import pickle
import os
import subprocess
import time
# %%
#parser = argparse.ArgumentParser(description="Script.")
#
#parser.add_argument("-i", "--idx", type=int, help="Category 0 (0-100 GeV) or 1 (100-160 GeV)", default=0)
#parser.add_argument('-v', "--variations", type=str, nargs='+', help="Nominal btag_Up btag_Down", default=None)
#parser.add_argument('-r', "--run", help="   Run 0 Performs saving of Std via bootstrap\
#                                            Run 1 Performs the full ABCD with Std saved", type=int, default=0)
#args = parser.parse_args()

#print("List of variations : ", args.variations)
idx, variations = 0, ['Nominal']#args.idx, args.variations



configFile = ["/t3home/gcelotto/ggHbb/abcd/new/configABCD.yaml",
               "/t3home/gcelotto/ggHbb/abcd/new/configABCD_100to160.yaml"][idx]
with open(configFile, "r") as f:
    config = yaml.safe_load(f)

dd = True
modelName = config["modelName"]
print(modelName)
print(variations)

dd = config["doubleDisco"]
x1 = config["x1"]
x2 = config["x2"]
xx = config["xx"]
t1 = config["t1"]
t2 = config["t2"]
isMCList = config["isMCList"]
isDataList = config["isDataList"]
detail = config["detail"]
if variations is None:
    variations = config["variations"]


outFolder, df_folder = "/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s"%modelName, "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/%s"%modelName
bins = np.load(outFolder+"/mass_bins.npy")
#midpoints = (bins[:-1] + bins[1:]) / 2
#bins = np.sort(np.concatenate([bins, midpoints]))
#midpoints = (bins[:-1] + bins[1:]) / 2
#bins = np.sort(np.concatenate([bins, midpoints]))
bins=np.array(bins, dtype=float)
# %%
dfs = []
dfProcessesMC, dfProcessesData, dfProcessesMC_JEC = getDfProcesses_v2()
# %%
# Loading data
dfsMC, dfsData, lumi = getDataFrames(dfProcessesMC, dfProcessesData, isMCList, isDataList, modelName, df_folder, dd)


dfsMC = cut(dfsMC, 'muon_pt', 9, None)
dfsData = cut(dfsData, 'muon_pt', 9, None)



# Visualize the score and choose where to put the cut
#plotNNscore(dfsData=dfsData)


# modify the name of saved plots
#for f in dfProcessesData.process[isDataList].values:
#    detail = detail + "_"+f[4:]
# %%
#Plot disCo for each dijetMass Bin
#dcor_data_values =  dcor_plot_Data(dfsData, dfProcessesData.process, isDataList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcor_%s_%s.png"%(modelName, detail), nEvents=90000)
#dcor_MC_values =  dcor_plot_Data(dfsMC, dfProcessesMC.process, isMCList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcorMC_%s_%s.png"%(modelName, detail), nEvents=-1)



# %%
for idx, df in enumerate(dfsMC):
    dfsMC[idx]['process'] = dfProcessesMC.iloc[isMCList].iloc[idx].process
dfsMC = cut(dfsMC, 'jet2_btagDeepFlavB', 0.71,  None)
dfsData = cut(dfsData, 'jet2_btagDeepFlavB', 0.71, None)
dfsMC = cut(dfsMC, 'jet1_btagDeepFlavB', 0.71,  None)
dfsData = cut(dfsData, 'jet1_btagDeepFlavB', 0.71, None)
dfsMC = cut(dfsMC, 'dijet_mass', 50,  None)
dfsData = cut(dfsData, 'dijet_mass', 50, None)
dfsMC = cut(dfsMC, 'PNN1', None,  0.92)
dfsData = cut(dfsData, 'PNN1', None, 0.92)
dfsMC = cut(dfsMC, 'PNN2', None,  0.92)
dfsData = cut(dfsData, 'PNN2', None, 0.92)


dfMC = pd.concat(dfsMC)
dfData = pd.concat(dfsData)
# %%
for b_low, b_high in zip(bins[:-1], bins[1:]):
    dfBin = dfData[(dfData.dijet_mass<b_high) & (dfData.dijet_mass>b_low)]
    t1_i, t1_f, t2_i, t2_f = 0.8, 0.85, 0., 0.8
    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #ax[0].hist(dfBin.PNN1, bins=31, histtype='step', density=True)
    #ax[1].hist(dfBin.PNN2, bins=31, histtype='step', density=True)
    #ax[0].hist(dfMC.PNN1, bins=31, histtype='step', density=True)
    #ax[1].hist(dfMC.PNN2, bins=31, histtype='step', density=True)
    #ax[0].vlines(x=[t1_f, t1_i], ymin=0, ymax=1, color='red')
    #ax[1].vlines(x=[t2_f, t2_i], ymin=0, ymax=1, color='red')
    #ax.hist(dfBin.PNN1, bins=31, histtype='step', density=True)

    #ax.hist(dfMC.PNN1, bins=31, histtype='step', density=True)



    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #ax[0].hist2d(dfBin.PNN1, dfBin.PNN2, bins=(100, 100))
    #ax[1].hist2d(dfMC.PNN1, dfMC.PNN2, bins=(100, 100))
    #ax[1].vlines(x=[t1_f, t1_i], ymin=0, ymax=1, color='red')
    #ax[1].hlines(y=[t2_f, t2_i], xmin=0, xmax=1, color='red')
    #print(pearsonr(dfBin.PNN1, dfBin.PNN2))




    mA = (dfBin.PNN1 < t1_i) & (dfBin.PNN2 > t2_f)
    mB = (dfBin.PNN1 > t1_i) & (dfBin.PNN1 < t1_f) & (dfBin.PNN2 > t2_f)
    mS = (dfBin.PNN1 > t1_f) & (dfBin.PNN2 > t2_f)
    mC = (dfBin.PNN1 < t1_i) & (dfBin.PNN2 < t2_f) & (dfBin.PNN2 > t2_i)
    mD = (dfBin.PNN1 > t1_i) & (dfBin.PNN1 < t1_f) & (dfBin.PNN2 < t2_f) & (dfBin.PNN2 > t2_i)
    mE = (dfBin.PNN1 > t1_f) & (dfBin.PNN2 < t2_f) & (dfBin.PNN2 > t2_i)
    mF = (dfBin.PNN1 < t1_i) & (dfBin.PNN2 < t2_i)
    mG = (dfBin.PNN1 > t1_i) & (dfBin.PNN1 < t1_f) & (dfBin.PNN2 < t2_i)
    mH = (dfBin.PNN1 > t1_f) & (dfBin.PNN2 < t2_i)


    A = len(dfBin[mA])
    B = len(dfBin[mB])
    C = len(dfBin[mC])
    D = len(dfBin[mD])
    E = len(dfBin[mE])
    F = len(dfBin[mF])
    G = len(dfBin[mG])
    H = len(dfBin[mH])
    S = len(dfBin[mS])
    expected = B*(E+H)/(D+G) * (A*(D+G)/((C+F)*B))**-1
    observed = S
    print(expected/observed)

    #mA = (dfBin.PNN1 < t1_f) & (dfBin.PNN2 > t2_f)
    #mB = (dfBin.PNN1 < t1_f) & (dfBin.PNN2 > t2_i) & (dfBin.PNN2 < t2_f)
    #mS = (dfBin.PNN1 > t1_f) & (dfBin.PNN2 > t2_f)
    #mC =  (dfBin.PNN1 > t1_f) & (dfBin.PNN2 > t2_i) & (dfBin.PNN2 < t2_f)
    #mD = (dfBin.PNN1 < t1_f) & (dfBin.PNN2 < t2_i)
    #mE = (dfBin.PNN1 > t1_f) & (dfBin.PNN2 < t2_i)
    #
    #A = len(dfBin[mA])
    #B = len(dfBin[mB])
    #S = len(dfBin[mS])
    #C = len(dfBin[mC])
    #D = len(dfBin[mD])
    #E = len(dfBin[mE])
#


    #expected = A*C/B * (B*E/(C*D))**(-1)
    #observed = S
    #print(expected/observed, " vs ", (A*(C+E)/(B+D))/observed)

# %%
'''
Analysis starts here!
'''



sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/helpers")


variation_map = {
    'Nominal': lambda df: df['weight'],
    'btag_Up': lambda df: df['weight'] * df['btag_up'] / df['btag_central'],
    'btag_Down': lambda df: df['weight'] * df['btag_down'] / df['btag_central'],
}

#variations_dijet_map = {
#    
#    f"{var}_{direction}": lambda df, v=var, d=direction: apply_dijet_variation(df, v, d)
#    for var in [
#    'JECAbsoluteMPFBias','JECAbsoluteScale','JECAbsoluteStat','JECFlavorQCD','JECFragmentation','JECPileUpDataMC','JECPileUpPtBB','JECPileUpPtEC1','JECPileUpPtEC2','JECPileUpPtHF','JECPileUpPtRef','JECRelativeBal','JECRelativeFSR','JECRelativeJEREC1','JECRelativeJEREC2','JECRelativeJERHF','JECRelativePtBB','JECRelativePtEC1','JECRelativePtEC2','JECRelativePtHF','JECRelativeSample','JECRelativeStatEC','JECRelativeStatFSR','JECRelativeStatHF','JECSinglePionECAL','JECSinglePionHCAL','JECTimePtEta',
#]
#    for direction in ["Up", "Down"]
#}
def apply_variation(df, variation):
    df['weight_'] = variation_map[variation](df)
    return df

x=(bins[1:]+bins[:-1])/2
chi2_mask = np.array(~((x > 100.6) & (x < 150.6 )), dtype=bool)

t1_i, t1_f, t2_i, t2_f =  0.07, 0.8, 0.17, 0.75
import matplotlib.pyplot as plt
import seaborn as sns

def plot_PNN_cut(df, t1_label, t2_label, t1_cut, t2_cut, detail, outFolder, dataset_label="Data" ):
    plt.figure(figsize=(8,6))
    plt.hist2d(df['PNN1'], df['PNN2'], bins=100, cmap='viridis', norm=plt.cm.colors.LogNorm())
    plt.colorbar(label='Counts')
    plt.axvline(t1_cut, color='red', linestyle='--', label=f'{t1_label} cut = {t1_cut}')
    plt.axhline(t2_cut, color='blue', linestyle='--', label=f'{t2_label} cut = {t2_cut}')
    plt.xlabel('PNN1')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('PNN2')
    plt.title(f'{detail}: {dataset_label}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outFolder)

# %%
for var in variations:
    print(var, "\n\n")
    dfMC_mod = dfMC.copy()
    # Apply variations to mjj or weight
    dfMC_mod = apply_variation(dfMC_mod, var)
    #print(dfMC_mod.dijet_mass, dfMC_mod.weight)
    detail = 'OOB'
    pulls_QCD_SR, err_pulls_QCD_SR = ABCD(dfData, dfMC_mod,  x1, x2, xx, bins, t1=0.8, t2=0.6, isMCList=isMCList, dfProcessesMC=dfProcessesMC, lumi=lumi, sameWidth_flag=False,
                                          suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail),
                                          blindPar=(True, 100.6, 150.6), chi2_mask=chi2_mask,
                                          #corrections=1/(pulls_QCD_SR1*pulls_QCD_SR2)**2,
                                          #err_corrections = np.sqrt((2*err_pulls_QCD_SR1/(pulls_QCD_SR1**3*pulls_QCD_SR2**2))**2  + (2*err_pulls_QCD_SR2/(pulls_QCD_SR2**3*pulls_QCD_SR1**2))**2)
                                          )


    # Five CRs Validation
    #detail = "V1"
    #dfData_Validation = dfData[(dfData.PNN1<t1_f)]
    #dfMC_mod_Validation = dfMC_mod[(dfMC_mod.PNN1<t1_f)]
    ##plot_PNN_cut(dfData_Validation, 't1_f', 't2_i', 0.8, 0.6, detail,outFolder="/t3home/gcelotto/ggHbb/abcd/new/plots/1.png", dataset_label="Data")
    #pulls_QCD_SR1, err_pulls_QCD_SR1 = ABCD(dfData_Validation, dfMC_mod_Validation,  x1, x2, xx, bins,
    #                                        t1=t1_i, t2=t2_f, isMCList=isMCList, dfProcessesMC=dfProcessesMC, lumi=lumi, sameWidth_flag=False,
    #                                      suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail),
    #                                      blindPar=(False, 100.6, 150.6), chi2_mask=np.full(len(bins)-1, True))
    #
    ## Five CRs SR
    #detail = "SR_5CR_noCorrection"
    #dfData_Validation = dfData[(dfData.PNN1>t1_i) ]
    #dfMC_mod_Validation = dfMC_mod[(dfMC_mod.PNN1>t1_i)]
#
    ##plot_PNN_cut(dfData_Validation, 't1_f', 't2_i', 0.8, 0.6, detail,outFolder="/t3home/gcelotto/ggHbb/abcd/new/plots/1.png", dataset_label="Data")
    #pulls_QCD, err_pulls_QCD = ABCD(dfData_Validation, dfMC_mod_Validation,  x1, x2, xx, bins, t1=t1_f, t2=t2_f, isMCList=isMCList, dfProcessesMC=dfProcessesMC, lumi=lumi, sameWidth_flag=False,
    #                                      suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail),
    #                                      blindPar=(False, 100.6, 150.6), chi2_mask=np.full(len(bins)-1, True),
    #                                      )
    #detail = "SR_5CR"
    #pulls_QCD, err_pulls_QCD = ABCD(dfData_Validation, dfMC_mod_Validation,  x1, x2, xx, bins, t1=t1_f, t2=t2_f, isMCList=isMCList, dfProcessesMC=dfProcessesMC, lumi=lumi, sameWidth_flag=False,
    #                                      suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail),
    #                                      blindPar=(False, 100.6, 150.6), chi2_mask=np.full(len(bins)-1, True),
    #                                      corrections = 1/pulls_QCD_SR1, err_corrections=err_pulls_QCD_SR1/pulls_QCD_SR1**2)
    #
    #
    ## Do the first ABCD estimation
    # Compute chi2 in the unblinded regions
    #detail = "Validation1"
    #dfData_Validation = dfData[(dfData.PNN1<t1_f) & (dfData.PNN2>t2_i) ] 
    #dfMC_mod_Validation = dfMC_mod[(dfMC_mod.PNN1<t1_f) & (dfMC_mod.PNN2>t2_i)] 
    #plot_PNN_cut(dfData_Validation, 't1_f', 't2_i', t1_f, t2_i, detail,outFolder="/t3home/gcelotto/ggHbb/abcd/new/plots/1.png", dataset_label="Data")
    #pulls_QCD_SR1, err_pulls_QCD_SR1 = ABCD(dfData_Validation, dfMC_mod_Validation,  x1, x2, xx, bins, t1=t1_i, t2=t2_f, isMCList=isMCList, dfProcessesMC=dfProcessesMC, lumi=lumi, sameWidth_flag=False,
    #                                      suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail),
    #                                      blindPar=(False, 100.6, 150.6), chi2_mask=np.full(len(bins)-1, True))
    #
    #
    #detail = "Validation2"
    #dfData_Validation = dfData[(dfData.PNN1>t1_i) & (dfData.PNN2<t2_f) ] 
    #dfMC_mod_Validation = dfMC_mod[(dfMC_mod.PNN1>t1_i) & (dfMC_mod.PNN2<t2_f)] 
    #plot_PNN_cut(dfData_Validation, 't1_f', 't2_i', t1_f, t2_i, detail,outFolder="/t3home/gcelotto/ggHbb/abcd/new/plots/2.png", dataset_label="Data")
    #pulls_QCD_SR2, err_pulls_QCD_SR2 = ABCD(dfData_Validation, dfMC_mod_Validation,  x1, x2, xx, bins, t1=t1_f, t2=t2_i, isMCList=isMCList, dfProcessesMC=dfProcessesMC, lumi=lumi, sameWidth_flag=False,
    #                                      suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail),
    #                                      blindPar=(False, 100.6, 150.6), chi2_mask=np.full(len(bins)-1, True))
#
    #
    #detail = "SR"
    #mData = ((dfData.PNN1<t1_i) & (dfData.PNN2>t2_f) |
    #         (dfData.PNN1>t1_f) & (dfData.PNN2>t2_f) |
    #         (dfData.PNN1<t1_i) & (dfData.PNN2<t2_i) |
    #         (dfData.PNN1>t1_f) & (dfData.PNN2<t2_i))
    #mMC  = ((dfMC_mod.PNN1<t1_i) & (dfMC_mod.PNN2>t2_f) |
    #         (dfMC_mod.PNN1>t1_f) & (dfMC_mod.PNN2>t2_f) |
    #         (dfMC_mod.PNN1<t1_i) & (dfMC_mod.PNN2<t2_i) |
    #         (dfMC_mod.PNN1>t1_f) & (dfMC_mod.PNN2<t2_i))
    #
    #dfData_SR   = dfData[mData] 
    #dfMC_mod_SR = dfMC_mod[mMC] 
    #plot_PNN_cut(dfData_SR, 't1_f', 't2_i', t1_f, t2_i, detail,outFolder="/t3home/gcelotto/ggHbb/abcd/new/plots/3.png", dataset_label="Data")
    #pulls_QCD_SR, err_pulls_QCD_SR = ABCD(dfData_SR, dfMC_mod_SR,  x1, x2, xx, bins, t1=0.5, t2=0.5, isMCList=isMCList, dfProcessesMC=dfProcessesMC, lumi=lumi, sameWidth_flag=False,
    #                                      suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail),
    #                                      blindPar=(True, 100.6, 150.6), chi2_mask=chi2_mask,
    #                                      corrections=1/(pulls_QCD_SR1*pulls_QCD_SR2)**2,
    #                                      err_corrections = np.sqrt((2*err_pulls_QCD_SR1/(pulls_QCD_SR1**3*pulls_QCD_SR2**2))**2  + (2*err_pulls_QCD_SR2/(pulls_QCD_SR2**3*pulls_QCD_SR1**2))**2)
    #                                      )
# Waiting for job finishing


# %%

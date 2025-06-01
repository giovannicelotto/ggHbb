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
# %%
parser = argparse.ArgumentParser(description="Script.")
try:
    parser.add_argument("-i", "--idx", type=int, help="Category 0 or 1", default=0)
    parser.add_argument('-v', "--variations", type=str, nargs='+', default=None)
    args = parser.parse_args()
    print(args.variations)
    idx=args.idx
    variations=args.variations
except:
    idx=0
    variations=None

#midpoints = (bins[:-1] + bins[1:]) / 2
#bins = np.sort(np.concatenate([bins, midpoints]))
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
dcor_data_values =  dcor_plot_Data(dfsData, dfProcessesData.process, isDataList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcor_%s_%s.png"%(modelName, detail), nEvents=90000)
#dcor_MC_values =  dcor_plot_Data(dfsMC, dfProcessesMC.process, isMCList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcorMC_%s_%s.png"%(modelName, detail), nEvents=-1)
# %%
for idx, df in enumerate(dfsMC):
    dfsMC[idx]['process'] = dfProcessesMC.iloc[isMCList].iloc[idx].process
dfMC = pd.concat(dfsMC)
dfData = pd.concat(dfsData)





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
chi2_mask = np.array(~((x > 70.6) & (x < 150.6 )), dtype=bool)

# %%
for var in variations:
    print(var, "\n\n")
    dfMC_mod = dfMC.copy()
    # Apply variations to mjj or weight
    dfMC_mod = apply_variation(dfMC_mod, var)
    #print(dfMC_mod.dijet_mass, dfMC_mod.weight)



    # Do the first ABCD estimation
    # Compute chi2 in the unblinded regions
    detail = config["detail"]+'_'+var
    pulls_QCD_SR, err_pulls_QCD_SR = ABCD(dfData, dfMC_mod,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=lumi, sameWidth_flag=False,
                                          suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail),
                                          blindPar=(True, 70.6, 150.6), chi2_mask=chi2_mask)

    #Compute pearson
    pearson_data_values, bootstrap_errors, confidence_intervals = runPearson(dfsData, bins, withErrors=False, num_bootstrap = 100, fraction = 1 / 5)

    # Make some plots
    maskBlind = makePlotsPearson(pulls_QCD_SR, err_pulls_QCD_SR, pearson_data_values, bins, bootstrap_errors)
    # Fit the model
    (m_fit, q_fit), (m_err, q_err), cov_matrix_fit = pullsVsDisco_pearson(pearson_data_values, pulls_QCD_SR, err_pulls_QCD_SR, xerr=bootstrap_errors,mask =chi2_mask, lumi=0, outName="/t3home/gcelotto/ggHbb/abcd/new/plots/pulls_vs_dcor/pulls_vs_dPearson_%s_%s.png"%(modelName, detail))


    #Extract correction
    corrections, err_corrections = computeCorrections(m_fit, m_err, q_fit, q_err, pearson_data_values, cov_matrix_fit)
    detailC = config["detail"]+'C_'+var

    newStds = getStdAllBins(dfData, dfMC, xx, bins, m_fit, q_fit, t1, t2, save=True, path = "/t3home/gcelotto/ggHbb/abcd/new/output/std_SR_b1_%s.npy"%var)
    newStds = np.array(newStds)

    chi2_maskNew = np.array(~((x > 100.5) & (x < 150)), dtype=bool) & ~chi2_mask
    #chi2_maskNew =  ~chi2_mask
    pulls_QCD_SR_new, err_pulls_QCD_SR_new = ABCD ( dfData, dfMC_mod,  x1, x2, xx, bins, t1, t2, isMCList,
                                                    dfProcessesMC, lumi=lumi,  blindPar=(True, 100.5, 150),
                                                    suffix='%s_%s'%(modelName, detailC),
                                                    sameWidth_flag=False,
                                                    corrections=corrections, err_corrections=newStds, var=var, chi2_mask=chi2_maskNew )
    plt.close('all')


# %%
print("New Uncertainties vs Old\n", err_pulls_QCD_SR_new/err_pulls_QCD_SR)

# %%
print("m : ", m_fit, " +- ", m_err)
print("q : ", q_fit, " +- ", q_err)
# %%

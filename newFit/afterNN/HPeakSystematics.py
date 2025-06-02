# %%
import numpy as np
import sys
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from helpers.allFunctions import *
from helpers.fitWithSystematics import *
import mplhep as hep
hep.style.use("CMS")
from functions import getDfProcesses_v2
import yaml
import argparse
# %%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, default=2, help='Index value (default: 2)')

    args = parser.parse_args()
    idx = args.idx
except:
    idx = 2
    print("Parser it not working.\n Idx set to %d"%idx)

config_path = ["/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/zPeakFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0/zPeakFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/zPeakFit_config.yml"][idx]

with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

MCList = config['MCList']
MCList_sD = config['MCList_sD']
MCList_sU = config['MCList_sU']
x1, x2 = config['x_bounds']
modelName = config['modelName']
outFolder = config['outFolder']
cuts_dict = config['cuts_dict']
fitFunction= config['fitFunction']
nbins = config["nbins"]
params = config["params"]
paramsLimits  = config["paramsLimits"]

path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
dfProcesses = getDfProcesses_v2()[0].iloc[MCList]
dfProcesses_sD = getDfProcesses_v2()[0].iloc[MCList_sD]
dfProcesses_sU = getDfProcesses_v2()[0].iloc[MCList_sU]

set_x_bounds(x1, x2)


# %%

fit_systematics = FitWithSystematics(modelName, path, dfProcesses, x1, x2, outFolder, fitFunction, dfProcesses_sD=dfProcesses_sD, dfProcesses_sU=dfProcesses_sU)
# %%
dfsMC = fit_systematics.load_data()
print("This are all the columns:")
print(dfsMC[0].columns)
print([len(dfsMC[i]) for i in range(len(dfsMC))])
dfsMC = fit_systematics.apply_cuts(dfsMC, cuts_dict)
bins = np.linspace(x1, x2, nbins)
x = (bins[1:] + bins[:-1]) / 2
cTot = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)
for idx, df in enumerate(dfsMC):
    c=np.histogram(df.dijet_mass, bins=bins, weights=df.weight)[0]
    cerr=np.histogram(df.dijet_mass, bins=bins, weights=(df.weight)**2)[0]
    err = err + cerr

    cTot=cTot+c
err = np.sqrt(err)
fitregion = ((x > x1) & (x < x2))
# %%
m = fit_systematics.fit_model(x, cTot, err, fitregion, params, paramsLimits)
print(m.values)
# %%
fit_systematics.save_parameters("nominal", m)
# %%
fit_systematics.plot_results(x, cTot, err, m, fitregion, bins, outFolder)
# %%
variations = [
    'btag_up', 'btag_down',
    #'JECAbsoluteMPFBias_Up', 'JECAbsoluteMPFBias_Down',
    #'JECAbsoluteScale_Up', 'JECAbsoluteScale_Down', 'JECAbsoluteStat_Up', 'JECAbsoluteStat_Down', 'JECFlavorQCD_Up', 'JECFlavorQCD_Down', 
    #'JECFragmentation_Up', 'JECFragmentation_Down', 'JECPileUpDataMC_Up', 'JECPileUpDataMC_Down', 'JECPileUpPtBB_Up', 'JECPileUpPtBB_Down',
    #'JECPileUpPtEC1_Up', 'JECPileUpPtEC1_Down', 'JECPileUpPtEC2_Up', 'JECPileUpPtEC2_Down', 'JECPileUpPtHF_Up', 'JECPileUpPtHF_Down', 'JECPileUpPtRef_Up',
    #'JECPileUpPtRef_Down', 'JECRelativeBal_Up', 'JECRelativeBal_Down', 'JECRelativeFSR_Up', 'JECRelativeFSR_Down', 'JECRelativeJEREC1_Up', 'JECRelativeJEREC1_Down',
    #'JECRelativeJEREC2_Up', 'JECRelativeJEREC2_Down', 'JECRelativeJERHF_Up', 'JECRelativeJERHF_Down','JECRelativePtBB_Up', 'JECRelativePtBB_Down', 'JECRelativePtEC1_Up',
    #'JECRelativePtEC1_Down', 'JECRelativePtEC2_Up', 'JECRelativePtEC2_Down', 'JECRelativePtHF_Up', 'JECRelativePtHF_Down', 'JECRelativeSample_Up', 'JECRelativeSample_Down',
    #'JECRelativeStatEC_Up', 'JECRelativeStatEC_Down', 'JECRelativeStatFSR_Up', 'JECRelativeStatFSR_Down', 'JECRelativeStatHF_Up', 'JECRelativeStatHF_Down',
    #'JECSinglePionECAL_Up', 'JECSinglePionECAL_Down', 'JECSinglePionHCAL_Up', 'JECSinglePionHCAL_Down', 'JECTimePtEta_Up', 'JECTimePtEta_Down',
    #'JER_Down', 'JER_Up'
    ]
fit_systematics.apply_variations(variations, dfsMC, bins, fitregion, params, paramsLimits, cuts_dict)
fit_systematics.plot_variations_results(x, cTot, err, bins, fitregion, outFolder)
fit_systematics.save_results()
# %%

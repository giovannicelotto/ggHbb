# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import getDfProcesses_v2
import yaml
import uproot
import glob, os
# %%
#isHiggsList = cfg['isHiggsList']
#isZList = cfg['isZList']
#isQCDList = cfg['isQCDList']
openingFile = open("/t3home/gcelotto/ggHbb/documentation/plotScripts/cutFlow_miniToNano.yaml", "r")
cfg = yaml.load(openingFile, Loader=yaml.FullLoader)
featuresForTraining = cfg['featuresForTraining']
grouped=cfg['grouped']
if grouped==1:
    processName_to_processNumber = cfg['processes_grouped']
else:
    processName_to_processNumber = cfg['processes_all']
# %%
processNames = list(processName_to_processNumber.keys())
nanoPaths = getDfProcesses_v2()[0].nanoPath


# %%
group_results = {}
group_results_unfiltered = {}
for processName in processNames:
    print("Processing group:", processName)

    total_xsection = 0.0

    processNumbers = processName_to_processNumber[processName]
    efficiencies_per_pN_weighted = []
    muon_filter_efficiencies_pN = []
    for processNumber in processNumbers:
        print("  Process number", processNumber)
        total_genWeight_pN = 0.0
        total_sumw_pN = 0.0

        path = nanoPaths.loc[processNumber]
        xsection_pN = getDfProcesses_v2()[0].loc[processNumber].xsection
        total_xsection += xsection_pN

        nanoFiles = glob.glob(os.path.join(path, "**", "*.root"), recursive=True)[:5]

        

        for idx, nanoFile in enumerate(nanoFiles):
            print("    File", idx + 1, "/", len(nanoFiles))

            with uproot.open(nanoFile) as file:
                branches = file["Events"].arrays()

                total_genWeight_pN += np.sum(branches["genWeight"].to_numpy())
                total_sumw_pN += file["Runs"].arrays()["genEventSumw"][0]



        efficiencies_per_pN_weighted.append(total_genWeight_pN * xsection_pN / (total_sumw_pN) if total_sumw_pN > 0 else 0.0)
        muon_efficiency_of_lastFile = sum(file['LuminosityBlocks'].arrays()['GenFilter_numEventsPassed'])/sum(file['LuminosityBlocks'].arrays()['GenFilter_numEventsTotal'])

        muon_filter_efficiencies_pN.append(muon_efficiency_of_lastFile)

    print(muon_filter_efficiencies_pN)
    group_results[processName] = {
        "ratio": np.sum(np.array(efficiencies_per_pN_weighted))/total_xsection
    }


    if sum(file['LuminosityBlocks'].arrays()['GenFilter_numEventsPassed'])/sum(file['LuminosityBlocks'].arrays()['GenFilter_numEventsTotal'])!=1:
        print("muon filter efficiency for process", processName, " equal to ", sum(file['LuminosityBlocks'].arrays()['GenFilter_numEventsPassed'])/sum(file['LuminosityBlocks'].arrays()['GenFilter_numEventsTotal']))
        group_results_unfiltered[processName] = {
            "ratio": np.sum(np.array(efficiencies_per_pN_weighted) * np.array(muon_filter_efficiencies_pN))/total_xsection
    }
    else:
        group_results_unfiltered[processName] = {
            "ratio": 0.0    
    }
    
    
    
    
labels = list(group_results.keys())
ratios = [group_results[p]["ratio"]*100 for p in labels]
ratios_unfiltered = [group_results_unfiltered[p]["ratio"]*100 for p in labels]
np.savez("/t3home/gcelotto/ggHbb/documentation/outputScripts_forPlotting/cutFlow_miniToNano_groupProcesses.npz", {"labels": labels, "ratios": ratios, "ratios_unfiltered": ratios_unfiltered})

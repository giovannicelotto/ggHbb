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
#load saved data
data = np.load("/t3home/gcelotto/ggHbb/documentation/plots/cutFlow_miniToNano_groupProcesses.npz", allow_pickle=True)["arr_0"].item()
# %%
labels = data["labels"]
ratios = data["ratios"]
ratios_unfiltered = data["ratios_unfiltered"]
# %%
fig, ax = plt.subplots(1, 1, figsize=(35, 10))

ax.bar(labels, ratios, label="Efficiency", align='center')
ax.set_ylabel(r" Passing Trigger / Generated [%]")
ax.set_xlabel("Process group")


visibilityFactor = 1
ax.bar(labels, np.array(ratios_unfiltered)*visibilityFactor, label="Efficiency x Muon Filter Efficiency", alpha=1, color='red', align='center')

for i, val in enumerate(ratios):
    if ratios_unfiltered[i]>0:
        ax.text(i,val + 1.5,"%.1f %%" % val,ha="center",va="bottom",fontsize=18,color="black")

        ax.text(i,val + 0," %.2f %%" % ratios_unfiltered[i],ha="center",va="bottom",fontsize=18,color="red")

    else:
        ax.text(i, val, "%.1f %%"%(val), ha="center", va="bottom", fontsize=18)
hep.cms.label(ax=ax, data=False)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.legend()
ax.set_ylim(0, max(ratios)*1.2)

if grouped==1:
    saveName = "cutFlow_miniToNano_groupProcesses.png"
else:
    saveName = "cutFlow_miniToNano_allProcesses.png"
outputFolder = "/t3home/gcelotto/ggHbb/documentation/plots/cutFlows/"
fig.savefig(outputFolder+saveName, bbox_inches="tight")
print("Plot saved to "+outputFolder+saveName)
# %%

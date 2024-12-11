
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import cut, loadMultiParquet, getDfProcesses
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
import glob
from hist import Hist


# %%
dfProcesses = getDfProcesses()
isMCList = [0, 1,
                2,
                3, 4, 5,
                6,7,8,9,10,11,
                12,13,14,
                15,16,17,18,19,
                20, 21, 22, 23, 36,
                #39    # Data2A
    ]
paths = dfProcesses.flatPath[isMCList]
paths[1] = paths[1]+"/others"
paths[0] = paths[0]+"/others"
# %%
nReal = 300
nMC = -1
columns = ['dijet_mass', 'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'sf', 'PU_SF']
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC,
                                                          columns=columns,
                                                          returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)

dfs = preprocessMultiClass(dfs=dfs)
for idx, df in enumerate(dfs):
    isMC = isMCList[idx]
    print("isMC ", isMC)
    print("Process ", dfProcesses.process[isMC])
    print("Xsection ", dfProcesses.xsection[isMC])
    dfs[idx]['weight'] = df.PU_SF*df.sf*dfProcesses.xsection[isMC] * nReal * 1000 * 0.774 /1017/numEventsList[idx]
    # make uinque data columns
if isMCList[-1]==39:
    dfs[0]=pd.concat([dfs[0], dfs[-1]])
# remove the last element (data2a)
    dfs = dfs[:-1]
#set to 1 weights of data
dfs[0]['weight'] = np.ones(len(dfs[0]))
# %%
bins = np.linspace(40, 300, 20)

retainedDataFraction = 0.2
cutsComboCollector = []
hA = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
hA.fill(dfs[0].dijet_mass)
QCDTotal= hA.copy()
# %%
for df, process in zip(dfs[1:], dfProcesses.process[isMCList][1:]):
                QCDTotal.fill(df.dijet_mass, weight=-df.weight)
                print(process, np.sum(np.histogram(df.dijet_mass, weights=df.weight)[0]))
            

# %%
for binIdx, (bmin, bmax) in enumerate(zip(bins[:-1], bins[1:])):
    cutsCombo = {
        "cut1": [],
        "cut2": [],
        "effBkg":[],
        "effSignal":[],
    }
    print(binIdx, bmin, " - ",bmax, " GeV")
    
    for cut1 in np.linspace(0, 1, 40):
        for cut2 in np.linspace(0, 1, 40):
            QCDRetained = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
            QCDRetained.fill(dfs[0].dijet_mass[(dfs[0].jet1_btagDeepFlavB > cut1) & (dfs[0].jet2_btagDeepFlavB > cut2)])
            dataRetained = QCDRetained.values()[binIdx]

            for df, process in zip(dfs[1:], dfProcesses.process[isMCList][1:]):
                #print(process)


                m = (df.dijet_mass > bmin) & (df.dijet_mass < bmax) & (df.jet1_btagDeepFlavB>cut1) & (df.jet2_btagDeepFlavB>cut2)
                #print("%.1f%% in %.1f - %.1f GeV"%(np.sum(m)/len(m)*100, bmin, bmax))
                QCDRetained.fill(df[m].dijet_mass, weight=-df[m].weight)
            

            if QCDRetained.values()[binIdx]/QCDTotal.values()[binIdx] < retainedDataFraction:
                print("QCD Retained ", QCDRetained.values()[binIdx]/QCDTotal.values()[binIdx]*100)
                numSignal = dfs[1].weight[(dfs[1].dijet_mass>bmin) & (dfs[1].dijet_mass<bmax) & (dfs[1].jet1_btagDeepFlavB>cut1) & (dfs[1].jet2_btagDeepFlavB>cut2)].sum()
                denSignal = dfs[1].weight[(dfs[1].dijet_mass>bmin) & (dfs[1].dijet_mass<bmax)].sum()
                print("Signal retained", numSignal/denSignal*100)

                cutsCombo["cut1"].append(cut1)
                cutsCombo["cut2"].append(cut2)
                cutsCombo["effBkg"].append(QCDRetained.values()[binIdx]/QCDTotal.values()[binIdx])
                cutsCombo["effSignal"].append(numSignal/denSignal)
                break
    cutsComboCollector.append(cutsCombo)
# %%
import json
with open('/t3home/gcelotto/ggHbb/abcd/dynamicABCD/cutsComboCollector.json', 'w') as f:
    json.dump(cutsComboCollector, f, indent=4)

# %%

ncol = 4
nrow = len(cutsComboCollector) // ncol + (len(cutsComboCollector) % ncol > 0)  # Ensure enough columns
fig, ax = plt.subplots(nrow, ncol, constrained_layout=True, figsize=(30, 20), sharex=True, sharey=True)
axes = ax.flat if nrow > 1 and ncol > 1 else [ax]
for i, dataset in enumerate(cutsComboCollector):
    ax = axes[i]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    scatter = ax.scatter(
            cutsComboCollector[i]["cut1"], 
            cutsComboCollector[i]["cut2"], 
            c=cutsComboCollector[i]["effSignal"],
            cmap="viridis",  
            s=50)
    maxIdx = np.argmax(cutsComboCollector[i]["effSignal"])
    ax.text(x=0.1, y=0.9, s="Signal Eff : %.1f%%"%(cutsComboCollector[i]["effSignal"][maxIdx]*100), fontsize=24)
    ax.text(x=0.1, y=0.82, s="Bkg Eff : %.1f%%"%(cutsComboCollector[i]["effBkg"][maxIdx]*100), fontsize=24)
    ax.plot(cutsComboCollector[i]["cut1"][maxIdx], cutsComboCollector[i]["cut2"][maxIdx], marker='x', color='red')
    ax.set_title("%.1f < mjj < %.1f"%(bins[i], bins[i+1]), fontsize=28)
    ax.vlines(x=cutsComboCollector[i]["cut1"][maxIdx], ymin=0, ymax=cutsComboCollector[i]["cut2"][maxIdx], color='black', linestyle='dotted')
    ax.hlines(y=cutsComboCollector[i]["cut2"][maxIdx], xmin=0, xmax=cutsComboCollector[i]["cut1"][maxIdx], color='black', linestyle='dotted')
    ax.tick_params(labelsize=24)
for j in range(len(cutsComboCollector), len(axes)):
    axes[j].axis("off")
fig.savefig("/t3home/gcelotto/ggHbb/abcd/dynamicABCD/scan.png")
# %%

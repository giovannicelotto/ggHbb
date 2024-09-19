# %%
import uproot
import numpy as np
import glob
import matplotlib.pyplot as plt
import awkward as ak
# %%
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Apr01/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/crab_TTTo2L2Nu/240401_133506/0000/*.root")[:2]
isoB  = []
Muon_ptB = []
Muon_pfRelIso03_allB = []
for fileName in fileNames:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    for ev in range(tree.num_entries):
        Muon_pfRelIso03_all = branches["Muon_pfRelIso03_all"][ev]
        nMuon = branches["nMuon"][ev]
        Muon_pt = branches["Muon_pt"][ev]

        isoB.append(np.sum((Muon_pt>20) & (Muon_pfRelIso03_all<0.01)))
    Muon_ptB = branches["Muon_pt"]
    Muon_pfRelIso03_allB = branches["Muon_pfRelIso03_all"]
# %%
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Mar05/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/240305_081723/0000/*.root")[:2]
isoS  = []
for fileName in fileNames:
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    for ev in range(tree.num_entries):
        Muon_pfRelIso03_all = branches["Muon_pfRelIso03_all"][ev]
        nMuon = branches["nMuon"][ev]
        Muon_pt = branches["Muon_pt"][ev]

        isoS.append(np.sum((Muon_pt>20) & (Muon_pfRelIso03_all<0.01)))
    Muon_ptS = branches["Muon_pt"]
    Muon_pfRelIso03_allS = branches["Muon_pfRelIso03_all"]
# %%
fig, ax =plt.subplots(1, 1)
ax.hist(ak.flatten(Muon_pfRelIso03_allS), bins=np.linspace(0, 4))


# %%
fig, ax = plt.subplots(1, 1)
bins = np.arange(10)
cB = np.histogram(isoB, bins=bins)[0]
cS = np.histogram(isoS, bins=bins)[0]
cB = cB/np.sum(cB)
cS = cS/np.sum(cS)

ax.hist(bins[:-1], bins=bins, weights=cB, label='isoB', histtype=u'step')
ax.hist(bins[:-1], bins=bins, weights=cS, label='isoS', histtype=u'step')
ax.legend()
# %%
fig, ax = plt.subplots(1, 1)
ax.scatter(ak.flatten(Muon_pfRelIso03_allB), ak.flatten(Muon_ptB), s=1)
ax.scatter(ak.flatten(Muon_pfRelIso03_allS), ak.flatten(Muon_ptS), s=1)
ax.set_xlim(0, 50)
ax.set_ylim(0, 150)

#ax.hist(bins[:-1], bins=bins, weights=cB, label='isoB', histtype=u'step')
#ax.hist(bins[:-1], bins=bins, weights=cS, label='isoS', histtype=u'step')
ax.legend()
# %%

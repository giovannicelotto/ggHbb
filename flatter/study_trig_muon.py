# %%
import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
# %%
from functions import getDfProcesses_v2
dfProcesses = getDfProcesses_v2()[0]
# %%
nanoPath = dfProcesses.iloc[37].nanoPath
import glob
files = glob.glob(nanoPath+"/*.root", recursive=True)
# %%
njets_with_muon = []
njets_with_muon7 = []
njets_with_muon7ip3 = []

pt_muons = []
for file in files[:10]:
    f = uproot.open(file)
    branches = f["Events"].arrays()
    for ev in range(f["Events"].num_entries):
        Jet_muonIdx1 = branches["Jet_muonIdx1"][ev]
        Muon_pt = branches["Muon_pt"][ev]
        njets_with_muon.append(ak.sum(Jet_muonIdx1>=0))

        nmu7=0
        nmu7ip3=0
        for muidx in Jet_muonIdx1[Jet_muonIdx1>=0]:
            pt_muons.append(Muon_pt[muidx])
            if Muon_pt[muidx]>7:
                nmu7+=1
                if abs(branches["Muon_dxy"][ev][muidx]/branches["Muon_dxyErr"][ev][muidx])>3:
                    nmu7ip3+=1
        njets_with_muon7.append(nmu7)
        njets_with_muon7ip3.append(nmu7ip3)

        #print("Jets with one muon:", ak.sum(Jet_muonIdx1>=0))

# %%
fig, ax = plt.subplots(1, 1)
c, b =np.histogram(njets_with_muon,  bins=np.arange(5))[:2]
cmu7, b =np.histogram(njets_with_muon7,  bins=np.arange(5))[:2]
cmuip3, b =np.histogram(njets_with_muon7ip3,  bins=np.arange(5))[:2]
c = c /np.sum(c)
cmu7 = cmu7 /np.sum(cmu7)
cmuip3 = cmuip3 /np.sum(cmuip3)
ax.errorbar(b[:-1], c,      xerr=0.5, yerr=np.sqrt(c * (1-c) / len(njets_with_muon)), label="n muon in jet",marker='o', linestyle='none')
ax.errorbar(b[:-1], cmu7,   xerr=0.5, yerr=np.sqrt(cmu7 * (1-cmu7) / len(njets_with_muon)), label="n muon7 in jet",marker='o', linestyle='none')
ax.errorbar(b[:-1], cmuip3, xerr=0.5, yerr=np.sqrt(cmuip3 * (1-cmuip3) / len(njets_with_muon)), label="n muon7ip3 in jet",marker='o', linestyle='none')

ax.text(x=0.05, y=0.95, s=f"{len(njets_with_muon)} events", transform=ax.transAxes, verticalalignment='top')
ax.set_ylim(0, 1)
ax.set_xlabel("Number of jets with muon")
ax.set_ylabel("Probability")
ax.legend()




# %%
fig, ax = plt.subplots(1, 1)
ax.hist(pt_muons, bins=np.linspace(0, 20, 21))
ax.set_xlabel("pt of muon")
# %%

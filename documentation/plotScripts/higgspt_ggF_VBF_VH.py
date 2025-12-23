# %%
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
# %%
file_url = "root://xrootd-cms.infn.it///store/mc/RunIISummer20UL18NanoAODv9/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/100000/099C5B27-C6D0-504C-A444-303C5FA46635.root"

file =  uproot.open(file_url)
tree = file['Events']
branches = tree.arrays(["GenPart_pdgId", "GenPart_pt", "GenPart_statusFlags", "genWeight"])
GenPart_pdgId       = branches["GenPart_pdgId"]
GenPart_pt          = branches["GenPart_pt"]
GenPart_statusFlags = branches["GenPart_statusFlags"]
genWeight = branches["genWeight"]
ggH_weight = np.array(genWeight)
ggH = np.array(GenPart_pt[(GenPart_pdgId==25) & (GenPart_statusFlags>8192) & (GenPart_statusFlags<16384)])
# %%
file_url = "root://xrootd-cms.infn.it///store/mc/RunIISummer20UL18NanoAODv9/VBFHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/8E4C21CD-774F-9D48-BC8D-159AA6E71FE2.root"
file =  uproot.open(file_url)
tree = file['Events']
branches = tree.arrays(["GenPart_pdgId", "GenPart_pt", "GenPart_statusFlags", "genWeight"])
GenPart_pdgId       = branches["GenPart_pdgId"]
GenPart_pt          = branches["GenPart_pt"]
GenPart_statusFlags = branches["GenPart_statusFlags"]
genWeight = branches["genWeight"]
VBF_weight = np.array(genWeight)
VBF = np.array(GenPart_pt[(GenPart_pdgId==25) & (GenPart_statusFlags>8192) & (GenPart_statusFlags<16384)])

file_url = "root://xrootd-cms.infn.it///store/mc/RunIISummer20UL18NanoAODv9/ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2820000/FE6AF237-7A43-C640-A0C0-672C18B80526.root"
file =  uproot.open(file_url)
tree = file['Events']
branches = tree.arrays(["GenPart_pdgId", "GenPart_pt", "GenPart_statusFlags", "genWeight"])
GenPart_pdgId       = branches["GenPart_pdgId"]
GenPart_pt          = branches["GenPart_pt"]
GenPart_statusFlags = branches["GenPart_statusFlags"]
genWeight = branches["genWeight"]
ZH_weight = np.array(genWeight)
ZH = np.array(GenPart_pt[(GenPart_pdgId==25) & (GenPart_statusFlags>8192) & (GenPart_statusFlags<16384)])

# %%
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 200, 101)
#ax.hist(np.clip(ggH, bins[0], bins[-1]), bins=bins, histtype='step', linewidth=2, label='ggH', density=True, weights=ggH_weight)
#ax.hist(np.clip(VBF, bins[0], bins[-1]), bins=bins, histtype='step', linewidth=2, label='VBF', density=True, weights=VBF_weight)
#ax.hist(np.clip(ZH, bins[0], bins[-1]), bins=bins, histtype='step', linewidth=2, label='ZH', density=True, weights=ZH_weight)
ax.hist(ggH, bins=bins, histtype='step', linewidth=2, label='ggF', density=True, weights=ggH_weight)
ax.hist(VBF, bins=bins, histtype='step', linewidth=2, label='VBF', density=True, weights=VBF_weight)
ax.hist(ZH, bins=bins, histtype='step', linewidth=2, label='ZH', density=True, weights=ZH_weight)
ax.set_yscale('log')
ax.set_xlabel("Higgs p$_T$ [GeV]")
ax.set_ylabel("Normalized Counts")

ax.legend(loc='upper right')
hep.cms.label()
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/Higgs_genStudies/ggH_VBF_VH.png", bbox_inches='tight')
# %%

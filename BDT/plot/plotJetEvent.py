# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import uproot
import numpy as np
# %%
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Mar05/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/240305_081723/0000/GluGluHToBB_Run2_mc_2024Mar05_190.root"
f = uproot.open(path)
tree = f['Events']
branches = tree.arrays()
# %%
ev = 3
Jet_pt      = branches["Jet_pt"][ev]
Jet_eta     = branches["Jet_eta"][ev]
Jet_phi     = branches["Jet_phi"][ev]
nJet        = branches["nJet"][ev]
Jet_genJetIdx = branches["Jet_genJetIdx"][ev]
GenJet_partonMotherPdgId = branches["GenJet_partonMotherPdgId"][ev]
fig, ax = plt.subplots(1, 1)
print(nJet)
for i in range(nJet):
    if Jet_genJetIdx[i]>-1:
        if GenJet_partonMotherPdgId[i]==25:
            color='green'
        else:
            color='black'
    else:
        color='black'
    #ax.scatter(Jet_eta[i], Jet_phi[i], s=100, color=color)
    circle = plt.Circle((Jet_eta[i], Jet_phi[i]), 0.4, color=color, alpha=0.4, edgecolor=color, linewidth=2)
    ax.add_artist(circle)
    print(Jet_eta[i], Jet_phi[i])

ax.set_xlim(-5, 5)
ax.set_ylim(-np.pi, np.pi)
ax.set_xlabel("$\eta$")
ax.set_ylabel(r"$\varphi$")

# %%

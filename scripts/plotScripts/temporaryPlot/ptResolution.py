import ROOT
import uproot, sys
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np

f = uproot.open("root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraph-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/270000/7340EA64-7191-164A-B21C-DB2E4F76676D.root")
tree = f['Events']
branches = tree.arrays()

jetpt=[]
correction=[]
genjetpt=[]


for ev in range(tree.num_entries):
    nJet = branches["nJet"][ev]
    Jet_pt = branches["Jet_pt"][ev]
    GenJet_pt = branches["GenJet_pt"][ev]
    Jet_genJetIdx = branches["Jet_genJetIdx"][ev]
    GenJet_partonFlavour = branches["GenJet_partonFlavour"][ev]
    Jet_partonFlavour = branches["Jet_partonFlavour"][ev]
    Jet_bRegCorr = branches["Jet_bRegCorr"][ev]
    Jet_eta = branches["Jet_eta"][ev]

    for jetIdx in range(nJet):
        if (Jet_genJetIdx[jetIdx]>=0) & (abs(Jet_eta[jetIdx]<2.5)):
            print(len(GenJet_pt), Jet_genJetIdx[jetIdx])



            if Jet_genJetIdx[jetIdx] < len(GenJet_pt):
                if abs(GenJet_partonFlavour[Jet_genJetIdx[jetIdx]])==5:
                    if Jet_pt[jetIdx]< 50:
                        jetpt.append(Jet_pt[jetIdx])
                        correction.append(Jet_bRegCorr[jetIdx])
                        genjetpt.append(GenJet_pt[Jet_genJetIdx[jetIdx]])



jetpt = np.array(jetpt)
correction = np.array(correction)
genjetpt = np.array(genjetpt)
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 2, 40)
counts_uncorrected = ax.hist(genjetpt/jetpt, bins=bins, histtype=u'step', label='uncorrected')[0]
counts_corrected = ax.hist(genjetpt/(jetpt*correction), bins=bins, histtype=u'step', label='corrected')[0]
ax.legend()
ax.set_xlabel("GenJet_pt/Jet_pt")
hep.cms.label(ax=ax)
fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/others/ptregression.png")
print("Saved in /t3home/gcelotto/ggHbb/outputs/plots/others/ptregression.png")



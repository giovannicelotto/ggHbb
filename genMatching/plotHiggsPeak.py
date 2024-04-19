import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import mplhep as hep
hep.style.use("CMS")

fig,ax = plt.subplots(1, 1)
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched/GluGlu*.parquet")
df = pd.read_parquet(fileNames)

bins = np.linspace(50, 300, 100)
m = (df.jet1_pt > 50) & (df.jet2_pt > 50) & (abs(df.jet1_eta)<2.5) & (abs(df.jet1_eta)<2.5)
ax.hist(df.dijet_mass[m], bins=bins, alpha=0.5, label='Uncorrected')
ax.hist(df.dijetCorr_mass[m], bins=bins, alpha=0.5, label='Corrected')
ax.set_ylim(ax.get_ylim())
ax.vlines(x=125.09, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label='125.09 GeV')
ax.set_xlabel("Dijet Mass [GeV]")
ax.legend()
fig.savefig("/t3home/gcelotto/ggHbb/genMatching/higgsPeak.png", bbox_inches='tight')

ax.clear()
bins = np.linspace(0, 2, 40)
counts_unc = ax.hist(pd.concat([df.genJet1_pt[m], df.genJet2_pt[m]]) / pd.concat([df.jet1_pt[m], df.jet2_pt[m]]), bins=bins, alpha=0.4, label='Uncorrected')[0]
counts_cor = ax.hist(pd.concat([df.genJet1_pt[m], df.genJet2_pt[m]]) / pd.concat([df.jet1Corr_pt[m], df.jet2Corr_pt[m]]), bins=bins, alpha=0.4, label='Corrected')[0]
ax.set_xlabel("GenJet_pt/Jet_pt")
ax.legend()
fig.savefig("/t3home/gcelotto/ggHbb/genMatching/ratio_Higgs.png", bbox_inches='tight')


# EWKZJets

fig,ax = plt.subplots(1, 1)
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched/EWKZJets*.parquet")
#print(fileNames)
df = pd.read_parquet(fileNames)
bins = np.linspace(20, 150, 40)
m = (df.jet1_pt > 50) & (df.jet2_pt > 50)
ax.hist(df.dijet_mass[m], bins=bins, alpha=0.5, label='Uncorrected')
ax.hist(df.dijetCorr_mass[m], bins=bins, alpha=0.5, label='Corrected')
ax.set_ylim(ax.get_ylim())
ax.vlines(x=91.18, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label='91.18 GeV')
ax.legend()
ax.set_xlabel("Dijet Mass [GeV]")
fig.savefig("/t3home/gcelotto/ggHbb/genMatching/ZPeak.png", bbox_inches='tight')

ax.clear()
bins = np.linspace(0, 2, 40)
counts_unc = ax.hist(pd.concat([df.genJet1_pt[m], df.genJet2_pt[m]]) / pd.concat([df.jet1_pt[m], df.jet2_pt[m]]), bins=bins, alpha=0.4, label='Uncorrected')[0]
counts_cor = ax.hist(pd.concat([df.genJet1_pt[m], df.genJet2_pt[m]]) / pd.concat([df.jet1Corr_pt[m], df.jet2Corr_pt[m]]), bins=bins, alpha=0.4, label='Corrected')[0]
ax.set_xlabel("GenJet_pt/Jet_pt")
ax.legend()
fig.savefig("/t3home/gcelotto/ggHbb/genMatching/ratio_Z.png", bbox_inches='tight')

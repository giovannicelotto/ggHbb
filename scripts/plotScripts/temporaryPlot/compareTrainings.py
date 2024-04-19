import numpy as np
import matplotlib.pyplot as plt
import glob
import mplhep as hep
hep.style.use("CMS")

oldFileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures/ggHbb_bScoreBased4_*.npy")
oldFileNames = oldFileNames[:30]
old = np.load(oldFileNames[0])
for file in oldFileNames[1:]:
    current = np.load(file)    
    old = np.vstack((old, current))
new = np.load("/t3home/gcelotto/ggHbb/outputs/ggHbb_bScoreBased4_3.npy")
std_old = np.std(old[old[:,21]<200,21])
std_new = np.std(new[new[:,21]<200,21])
print(old.shape, new.shape)
fig, ax = plt.subplots(1, 1)
ax.hist(old[:,21], bins=np.linspace( 0, 500, 100), color='blue', histtype=u'step', label='old (std:%.1f)'%std_old, density=True)
ax.hist(new[:,21], bins=np.linspace( 0, 500, 100), color='red', histtype=u'step', label='new (std:%.1f)'%std_new, density=True)
ax.set_ylim(0, ax.get_ylim()[1])
ax.vlines(x=125.09, ymin=0, ymax=ax.get_ylim()[1], label='125.09', color='black', linestyles='---')
ax.legend()
ax.set_xlabel("Dijet Mass [GeV]")
fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/compareTraining.pdf", bbox_inches='tight')

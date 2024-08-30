import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import mplhep as hep
hep.style.use("CMS")
pathToZ ="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets"
lowerLimit = [100, 200, 400, 600, 800]
dfs=[]
for ll in lowerLimit:
    fileNames = glob.glob(pathToZ+"/ZJetsToQQ_HT-%d*/*.parquet"%ll)
    df = pd.read_parquet(fileNames)
    dfs.append(df)

fig, ax = plt.subplots(1, 1)
bins = np.linspace(50, 130, 80)
labels = ['ZJets_HT100-200', 'ZJets_HT200-400', 'ZJets_HT400-600', 'ZJets_HT600-800', 'ZJets_HT800-Inf']
for idx, df in enumerate(dfs):
    counts = np.histogram(df.dijet_mass, bins=bins, weights=df.sf)[0]
    counts=counts/np.sum(counts)
    ax.hist(bins[:-1], bins=bins, weights=counts, label=labels[idx], histtype=u'step')
hep.cms.label()
ax.legend()
outName = '/t3home/gcelotto/ggHbb/outputs/plots/massSpectrum/ZPeak_HTClass.png'
print("Saved ", outName)
fig.savefig(outName)
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functions import cut
import glob
# %%
pathqq = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJetsqq"
pathbb = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJetsBB"
pathdata = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others/Data_99*.parquet")
dfqq = pd.read_parquet(pathqq)
dfbb = pd.read_parquet(pathbb)
dfdata = pd.read_parquet(pathdata)
dfbb, dfqq, dfdata = cut([dfbb, dfqq, dfdata], 'jet1_pt', 20, None)
dfbb, dfqq, dfdata = cut([dfbb, dfqq, dfdata], 'jet2_pt', 20, None)



# %%
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 1, 3)
cqq = np.histogram((dfqq.jet1_btagDeepFlavB>0.3) & (dfqq.jet2_btagDeepFlavB>0.3),bins=bins)[0]
cbb = np.histogram((dfbb.jet1_btagDeepFlavB>0.3) & (dfbb.jet2_btagDeepFlavB>0.3),bins=bins)[0]
cqq=cqq/np.sum(cqq)
cbb=cbb/np.sum(cbb)
ax.hist(bins[:-1], bins=bins, weights=cqq, histtype='step', label='qq')
ax.hist(bins[:-1], bins=bins, weights=cbb, histtype='step', label='bb')
ax.legend()

# %%
dfbb, dfqq, dfdata = cut([dfbb, dfqq, dfdata], 'jet1_btagDeepFlavB', 0.3, None)
dfbb, dfqq, dfdata = cut([dfbb, dfqq, dfdata], 'jet2_btagDeepFlavB', 0.3, None)

# %%
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0.3, 1, 11)
cqq = np.histogram(np.clip(dfqq.jet2_btagDeepFlavB, bins[0], bins[-1])      ,bins=bins)[0]
cbb = np.histogram(np.clip(dfbb.jet2_btagDeepFlavB, bins[0], bins[-1])      ,bins=bins)[0]
cdata = np.histogram(np.clip(dfdata.jet2_btagDeepFlavB, bins[0], bins[-1])  ,bins=bins)[0]

cqq=cqq/np.sum(cqq)
cbb=cbb/np.sum(cbb)
cdata=cdata/np.sum(cdata)

ax.hist(bins[:-1], bins=bins, weights=cqq, histtype='step', label='qq')
ax.hist(bins[:-1], bins=bins, weights=cbb, histtype='step', label='bb')
ax.hist(bins[:-1], bins=bins, weights=cdata, histtype='step', label='data')
ax.legend()
# %%

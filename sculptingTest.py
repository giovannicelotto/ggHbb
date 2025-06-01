# %%
from functions import *
import pandas as pd
import matplotlib.pyplot as plt
# %%
dfs = loadMultiParquet_Data_new(dataTaking=[2], nReals=1500, columns=None, filters=getCommonFilters(btagTight=True))
# %%
df = pd.concat(dfs[0])
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(df.dijet_mass[(df.dijet_pt>100) & (df.nJets==2)], bins=np.linspace(40, 200, 61), density=True, histtype='step', label='nJets = 2')
ax.hist(df.dijet_mass[(df.dijet_pt>100) & (df.nJets==3)], bins=np.linspace(40, 200, 61), density=True, histtype='step', label='nJets = 3')
ax.hist(df.dijet_mass[(df.dijet_pt>100) & (df.nJets==4)], bins=np.linspace(40, 200, 61), density=True, histtype='step', label='nJets = 4')
ax.hist(df.dijet_mass[(df.dijet_pt>100) & (df.nJets>=5)], bins=np.linspace(40, 200, 61), density=True, histtype='step', label='nJets = 5')
#ax.hist(df.dijet_mass[(df.dijet_pt>100) & (df.nJets>=7)], bins=np.linspace(40, 200, 61), density=True, histtype='step', label='nJets = 7+')
ax.legend()
# %%

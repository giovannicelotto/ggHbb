# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
wtrain = np.load("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/rWtrain.npy")
ytrain = np.load("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/Ytrain.npy")
xtrain = pd.read_parquet("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/Xtrain.parquet" )
genMassTrain = np.load("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/genMassTrain.npy")
# %%
print(wtrain[genMassTrain==0].mean())
print(wtrain[genMassTrain==125].mean())
# %%
fig, ax = plt.subplots(1, 1)
bins=np.linspace(0, 10, 101)
ax.hist(wtrain[genMassTrain==0], histtype='step', bins=bins, label='bkg')
ax.hist(wtrain[genMassTrain==125], histtype='step', bins=bins, label='signal')
ax.text(x=0.95, y=0.6, s="Mean Signal %.2f"%(wtrain[genMassTrain==125].mean()), transform=ax.transAxes, ha='right')
ax.text(x=0.95, y=0.55, s="Mean Bkg %.2f"%(wtrain[genMassTrain==0].mean()), transform=ax.transAxes, ha='right')
ax.text(x=0.95, y=0.4, s="Sum Signal  %.2f"%(wtrain[genMassTrain==125].sum()), transform=ax.transAxes, ha='right')
ax.text(x=0.95, y=0.35, s="Sum Bkg %.2f"%(wtrain[genMassTrain==0].sum()), transform=ax.transAxes, ha='right')
ax.legend()
ax.set_xlabel("weights")
# %%

# %%
from functions import loadMultiParquet_Data_new, getCommonFilters

# %%
dfs = loadMultiParquet_Data_new(dataTaking=[0], nReals=300, filters=getCommonFilters(btagWP="M", cutDijet=False, ttbarCR=0))
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = dfs[0][0]
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(df.dijet_dPhi, bins=np.linspace(-3.14, 3.14, 201))

# %%
fig, ax = plt.subplots(1, 1)
ax.hist(df.dijet_pT_asymmetry, bins=np.linspace(-3.14, 3.14, 201))
# %%
fig, ax = plt.subplots(1, 1)

bins = np.linspace(40, 180, 201)
# define masks
mask_base = np.ones(len(df), dtype=bool)
mask_pt_low =  (df.dijet_pt>60) & (df.dijet_pt < 120)
mask_pt_high = (df.dijet_pt>60) & ( df.dijet_pt > 120)
#mask_dphi = np.abs(df.dijet_dPhi) > 1.
asym_value = 0.45
mask_asym = np.abs(df.dijet_pT_asymmetry) < asym_value

# if you have it
mask_deta = np.abs(df.jet1_eta - df.jet2_eta) < 1.5


# baseline
ax.hist(df.loc[mask_base, 'dijet_mass'], bins=bins, histtype='step',
        density=True, label='no cuts')
# pt
ax.hist(df.loc[mask_pt_high, 'dijet_mass'], bins=bins, histtype='step',
        density=True, label='pt > 120')
ax.hist(df.loc[mask_pt_low, 'dijet_mass'], bins=bins, histtype='step',
        density=True, label='60 < pt < 120')
## Δφ cut
#ax.hist(df.loc[mask_dphi & mask_pt, 'dijet_mass'], bins=bins, histtype='step',
#        density=True, label='|Δφ| > 1.')
#
# + asymmetry
mask_2 = mask_pt_low & mask_asym
ax.hist(df.loc[mask_2, 'dijet_mass'], bins=bins, histtype='step',
        density=True, label='low pT+ asym < %.2f'%asym_value)
ax.set_yscale('log')

ax.set_xlabel('mjj')
ax.set_ylabel('Normalized entries')
ax.legend()

plt.show()
# %%
fig, ax  =plt.subplots(1, 1)
ax.hist(df.loc[mask_2].dijet_pT_asymmetry)
# %%

plt.hist(df.dijet_pT_asymmetry[(df.dijet_pt<120) & (df.dijet_pt>60)], bins=np.linspace(-1, 1, 101))
# %%

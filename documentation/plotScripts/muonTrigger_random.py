# %% 
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import uproot
import glob
import pandas as pd
import awkward as ak
# %%
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data50_Inf2026Apr06"
files = glob.glob(f"{path}/**/*.root", recursive=True)
# %%
# Open with uproot randomly from files 100 files and plot Muon_pt branch
muon_pts = []
muon_ipsig = []
for idx, file in enumerate(np.random.choice(files, size=50, replace=False)):
    print(idx)
    with uproot.open(file) as f:
        tree = f["Events"]
        branches = tree.arrays()
        Muon_pt = branches["Muon_pt"]
        Muon_dxy = branches["Muon_dxy"]
        Muon_dxyErr = branches["Muon_dxyErr"]
        Muon_isTriggering = branches["Muon_isTriggering"]


        muon_pts.append(Muon_pt[ak.sum(Muon_isTriggering, axis=1)>=1,0])
        muon_ipsig.append(Muon_dxy[ak.sum(Muon_isTriggering, axis=1)>=1,0]/Muon_dxyErr[ak.sum(Muon_isTriggering, axis=1)>=1,0])
# %%
muon_pts_ = np.concatenate(muon_pts)
muon_ipsig_ = np.concatenate(muon_ipsig)
# %%
# Plot histogram of muon_pts
fig, ax = plt.subplots(1, 1)
ax.hist(muon_pts_, bins=50, range=(0, 20), histtype="step", label="Muon_pt")
ax.set_xlabel("Muon pt [GeV]")
ax.set_ylabel("Events")
ax.text(x=0.9, y=0.9, s="Muon pt sampling 100 files", transform=ax.transAxes, ha="right", va="top") 
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(abs(muon_ipsig_), bins=1500, range=(0, 20), histtype="step", label="Muon_pt")
ax.set_xlabel("Muon IPsig")
ax.set_ylabel("Events")
ax.text(x=0.9, y=0.9, s="Muon IPsig sampling 100 files", transform=ax.transAxes, ha="right", va="top") 
# %%














path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
files = glob.glob(f"{path}/Data*/*.parquet", recursive=True)
# %%
# Open with uproot randomly from files 100 files and plot Muon_pt branch
muon_pts = []
muon_ipsig = []
for idx, file in enumerate(np.random.choice(files, size=1500, replace=False)):
    print(idx)
    df = pd.read_parquet(file)
    muon_pts.append(df["jet1_muon_pt"].values)
    muon_ipsig.append(abs(df["jet1_muon_dxySig"].values))

# %%
muon_pts_ = np.concatenate(muon_pts)
muon_ipsig_ = np.concatenate(muon_ipsig)
# %%
# Plot histogram of muon_pts
fig, ax = plt.subplots(1, 1)
ax.hist(muon_pts_, bins=1000, range=(0, 20), histtype="step", label="Muon_pt")
ax.set_xlabel("Muon pt [GeV]")
ax.set_ylabel("Events")
ax.text(x=0.9, y=0.9, s="Muon pt sampling 1500 files", transform=ax.transAxes, ha="right", va="top") 

fig, ax = plt.subplots(1, 1)
ax.hist(abs(muon_ipsig_), bins=501, range=(0, 20), histtype="step", label="Muon_pt")
ax.set_xlabel("Muon IPsig")
ax.set_ylabel("Events")
ax.text(x=0.9, y=0.9, s="Muon IPsig sampling 1500 files", transform=ax.transAxes, ha="right", va="top") 
# %%

gain = ((abs(muon_ipsig_)<6) & (abs(muon_ipsig_)>4) & (muon_pts_>9))
print(f"Gain in events with 4<IPsig<6 : {np.sum(gain)/np.sum(muon_pts_>9)*100:.2f} %")




# %%

























from functions import loadMultiParquet_v2

dfs, sumw = loadMultiParquet_v2(paths=[37], nMCs=-1, returnNumEventsTotal=True, filters=[('dijet_pt', '>', 50)])
for idx, df in enumerate(dfs):
    dfs[idx]['weight'] = df.flat_weight * df.xsection / sumw[idx]

# %%
df = pd.concat(dfs)

# %%
# Plot histogram of muon_pts

fig, ax = plt.subplots(1, 1)
ax.hist(abs(df.jet1_muon_pt), bins=100, range=(0, 20), histtype="step", label="Muon_IPsig", weights = df.weight)
ax.set_xlabel("Muon pt")
ax.set_ylabel("Events")
ax.text(x=0.9, y=0.9, s=f"Muon pt sampling ", transform=ax.transAxes, ha="right", va="top") 
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(abs(df.jet1_muon_dxySig), bins=100, range=(0, 20), histtype="step", label="Muon_IPsig", weights = df.weight)
ax.set_xlabel("Muon IPsig")
ax.set_ylabel("Events")
# %%

gain = ((abs(muon_ipsig_)<6) & (abs(muon_ipsig_)>4) & (muon_pts_>9))
print(f"Gain in events with 4<IPsig<6 : {np.sum(gain)/np.sum(muon_pts_>9)*100:.2f} %")







# %%

from functions import loadMultiParquet_v2
qcd_list = list(range(23,35))
dfs, sumw = loadMultiParquet_v2(paths=qcd_list, nMCs=20, returnNumEventsTotal=True, filters=[('dijet_pt', '>', 50)])
for idx, df in enumerate(dfs):
    dfs[idx]['weight'] = df.flat_weight * df.xsection / sumw[idx]

# %%
df = pd.concat(dfs)

# %%
# Plot histogram of muon_pts

fig, ax = plt.subplots(1, 1)
ax.hist(abs(df.jet1_muon_pt), bins=200, range=(0, 20), histtype="step", label="Muon_IPsig", weights = df.weight)
ax.set_xlabel("Muon pt")
ax.set_ylabel("Events")
ax.text(x=0.9, y=0.9, s=f"Muon pt sampling ", transform=ax.transAxes, ha="right", va="top") 
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(abs(df.jet1_muon_dxySig), bins=100, range=(0, 20), histtype="step", label="Muon_IPsig", weights = df.weight)
ax.set_xlabel("Muon IPsig")
ax.set_ylabel("Events")
#ax.text(x=0.9, y=0.9, s=f"Muon IPsig sampling  files", transform=ax.transAxes, ha="right", va="top") 
# %%

gain = ((abs(muon_ipsig_)<6) & (abs(muon_ipsig_)>4) & (muon_pts_>9))
print(f"Gain in events with 4<IPsig<6 : {np.sum(gain)/np.sum(muon_pts_>9)*100:.2f} %")






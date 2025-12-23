# %%
import matplotlib.pyplot as plt
import uproot
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import glob
import pandas as pd
# %%
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/DataPt302025Dec01/ParkingBPH1/crab_data_Run2018D_part1/251201_172318/0000"
fileNames = glob.glob(f"{path}/*.root")
# %%

# Branches you want to extract
branches_to_keep = ["dijet_mass", "dijet_pt"]  # replace with your actual branch names

# List to accumulate dataframes
df_list = []

for fname in fileNames[:500]:
    with uproot.open(fname) as f:
        tree = f["Events"]  # adjust if tree has a different name
        # Read only selected branches
        df = tree.arrays(branches_to_keep, library="pd")
        df_list.append(df)

# Concatenate all files
df = pd.concat(df_list, ignore_index=True)

# Save to Parquet (fast, preserves column types)
#df.to_parquet("selected_branches.parquet")

#print(f"Saved {len(df)} events with branches {branches_to_keep} to 'selected_branches.parquet'")

# %%
fig, ax = plt.subplots(1,1)
ax.hist(df.dijet_mass, bins=100, range=(50, 200), histtype='step', label='Data (p$_T>30$ GeV)', density=True)
ax.hist(df.dijet_mass[df.dijet_pt>50], bins=100, range=(50, 200), histtype='step', label='Data (p$_T>50$ GeV)', density=True)
ax.hist(df.dijet_mass[df.dijet_pt>70], bins=100, range=(50, 200), histtype='step', label='Data (p$_T>70$ GeV)', density=True)
ax.hist(df.dijet_mass[df.dijet_pt>100], bins=100, range=(50, 200), histtype='step', label='Data (p$_T>100$ GeV)', density=True)
ax.legend()
ax.set_xlabel("Dijet mass [GeV]")
ax.set_ylabel("Normalized counts")
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/dijet_mass_smooth_data_pt_cuts.png", bbox_inches="tight")
# %%

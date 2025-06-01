# %%
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
# %%
fileNames =glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/others/*.parquet")
df = pd.read_parquet(fileNames)


#df = df[df.dijet_pt>100]
# Compute total matching score for both hypotheses
score_12 = df.dR_b_jet1  + df.dR_antib_jet2 
score_21 = df.dR_b_jet2  + df.dR_antib_jet1 

# Select best match
use_12 = score_12 < score_21
use_21 = ~use_12

# Now, assign jet1/jet2 to b and antib depending on best matching
jet1_dR = np.where(use_12, df.dR_b_jet1, df.dR_antib_jet1)
jet1_dpT = np.where(use_12, df.dpT_pT_b_jet1, df.dpT_pT_antib_jet1)
jet1_label = np.where(use_12, 'b', 'antib')  # jet1 always matched to b

jet2_dR = np.where(use_12, df.dR_antib_jet2, df.dR_b_jet2)
jet2_dpT = np.where(use_12, df.dpT_pT_antib_jet2, df.dpT_pT_b_jet2)
jet2_label = np.where(use_12, 'antib', 'b')  # jet2 always matched to antib

# %%
# Matching conditions
match_jet1 = (jet1_dR < 0.2) & (jet1_dpT < 0.5)
match_jet2 = (jet2_dR < 0.2) & (jet2_dpT < 0.5)
match_both = match_jet1 & match_jet2

# %%
n_total = len(df)
n_jet1 = np.sum(match_jet1)
n_jet2 = np.sum(match_jet2)
n_both = np.sum(match_both)

print(f"Jet1 matched correctly in {n_jet1}/{n_total} events ({100*n_jet1/n_total:.1f}%)")
print(f"Jet2 matched correctly in {n_jet2}/{n_total} events ({100*n_jet2/n_total:.1f}%)")
print(f"Both matched correctly in {n_both}/{n_total} events ({100*n_both/n_total:.1f}%)")

# %%

import matplotlib.gridspec as gridspec
import numpy as np

# Bin settings
x_bins = np.linspace(0, 0.4, 31)
y_bins = np.linspace(-1, 1, 31)

# Set up figure and layout
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       wspace=0.05, hspace=0.05)

# Axes
ax_main = plt.subplot(gs[1, 0])  # 2D hist
ax_x = plt.subplot(gs[0, 0], sharex=ax_main)  # Top
ax_y = plt.subplot(gs[1, 1], sharey=ax_main)  # Right

# Main 2D histogram
h = ax_main.hist2d(jet1_dR, jet1_dpT, bins=(x_bins, y_bins), cmap='viridis')
fig.colorbar(h[3], ax=ax_main)
ax_main.set_xlabel('jet1 ΔR')
ax_main.set_ylabel('jet1 ΔpT/pT')

# Marginals
ax_x.hist(jet1_dR, bins=x_bins, color='gray')
ax_y.hist(jet1_dpT, bins=y_bins, orientation='horizontal', color='gray')

# Hide tick labels on marginals
plt.setp(ax_x.get_xticklabels(), visible=False)
plt.setp(ax_y.get_yticklabels(), visible=False)

# Tight layout
plt.tight_layout()
plt.show()










# The same for jet2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Bin settings
x_bins = np.linspace(0, 0.4, 31)
y_bins = np.linspace(-1, 1, 31)

# Set up figure and layout
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       wspace=0.05, hspace=0.05)

# Axes
ax_main = plt.subplot(gs[1, 0])  # 2D hist
ax_x = plt.subplot(gs[0, 0], sharex=ax_main)  # Top
ax_y = plt.subplot(gs[1, 1], sharey=ax_main)  # Right

# Main 2D histogram
h = ax_main.hist2d(jet2_dR, jet2_dpT, bins=(x_bins, y_bins), cmap='viridis')
fig.colorbar(h[3], ax=ax_main)
ax_main.set_xlabel('jet2 ΔR')
ax_main.set_ylabel('jet2 ΔpT/pT')

# Marginals
ax_x.hist(jet2_dR, bins=x_bins, color='gray')
ax_y.hist(jet2_dpT, bins=y_bins, orientation='horizontal', color='gray')

# Hide tick labels on marginals
plt.setp(ax_x.get_xticklabels(), visible=False)
plt.setp(ax_y.get_yticklabels(), visible=False)

# Tight layout
plt.tight_layout()
plt.show()











# %%
# Plot heatmap to scan in efficiency
dR_cuts = np.linspace(0.05, 0.5, 20)
dpT_cuts = np.linspace(0.05, 0.5, 20)

eff_both = np.zeros((len(dR_cuts), len(dpT_cuts)))

for i, dR_cut in enumerate(dR_cuts):
    for j, dpT_cut in enumerate(dpT_cuts):
        match_jet1 = (jet1_dR < dR_cut) & (jet1_dpT < dpT_cut)
        match_jet2 = (jet2_dR < dR_cut) & (jet2_dpT < dpT_cut)
        match_both = match_jet1 & match_jet2
        eff_both[i, j] = np.sum(match_both) / n_total


fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(eff_both.T, origin='lower', extent=[
    dR_cuts[0], dR_cuts[-1], dpT_cuts[0], dpT_cuts[-1]
], aspect='auto', cmap='viridis')

ax.set_xlabel('ΔR Cut')
ax.set_ylabel('ΔpT/pT Cut')
ax.set_title('Efficiency of Matching Both Jets')
fig.colorbar(im, ax=ax, label='Efficiency')

# %%

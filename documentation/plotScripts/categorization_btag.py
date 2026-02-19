# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# %%
folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Jan21_3_50p0"
print("Opening Signal")
df = pd.read_parquet(f"{folder}/df_GluGluHToBBMINLO_Jan21_3_50p0.parquet")
print("Opening Data")
df_data = pd.read_parquet([f"{folder}/dataframes_Data2D_Jan21_3_50p0.parquet",
                           #f"{folder}/dataframes_Data3D_Jan21_3_50p0.parquet",
                           #f"{folder}/dataframes_Data4D_Jan21_3_50p0.parquet",
                           ], columns=["jet1_btagDeepFlavB", "jet2_btagDeepFlavB","PNN", "dijet_mass", "dijet_pt" ])
# %%
#df_data = df_data.iloc[:]
#df =df[ ~((df.jet1_btagDeepFlavB>0.71)&(df.jet2_btagDeepFlavB>0.71)) ] 
#df_data =df_data[ ~((df_data.jet1_btagDeepFlavB>0.71)&(df_data.jet2_btagDeepFlavB>0.71)) ] 
df =df[(df.PNN> 0.8) & (df.PNN < 0.9) ] 
df_data =df_data[ (df_data.PNN>0.8)  &(df_data.PNN<0.9)] 
bins = np.linspace(0, 1, 21)
fig, ax = plt.subplots(1, 1)
#c=ax.hist(df.PNN, bins=bins, density=False, histtype='step', label='Signal', color='red', weights=df.flat_weight*41.6)[0]
#cdata=ax.hist(df_data.PNN, bins=bins, density=False, histtype='step', label='Signal', color='blue')[0]
c=np.histogram(df.PNN, bins=bins, density=False, weights=df.flat_weight*41.6)[0]
cdata=np.histogram(df_data.PNN, bins=bins, density=False, )[0]
ax.legend()
ax.set_yscale('log')
ax.plot((bins[1:]+bins[:-1])/2, c/np.sqrt(cdata))
#ax.plot((bins[1:]+bins[:-1])/2, c/cdata)

#df_data =df_data[(df_data.PNN>0.95) ]
#df =df[ (df.PNN>0.8) & (df.PNN<0.9) ] 
#df_data =df_data[(df_data.PNN>0.8) & ((df_data.PNN<0.9)) ]

# %%


# variables
x = df["jet1_btagDeepFlavB"].values
y = df["jet2_btagDeepFlavB"].values

# stack for KDE
values = np.vstack([x, y])
kde = gaussian_kde(values)

# evaluation grid (zoomed range is crucial)
xmin, xmax = 0.0, 1
ymin, ymax = 0.0, 1

xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 101),
    np.linspace(ymin, ymax, 101)
)

positions = np.vstack([xx.ravel(), yy.ravel()])
zz = kde(positions).reshape(xx.shape)



x_ = df_data["jet1_btagDeepFlavB"].values
y_ = df_data["jet2_btagDeepFlavB"].values

# stack for KDE
values_ = np.vstack([x_, y_])
kde = gaussian_kde(values_)


xx_, yy_ = np.meshgrid(
    np.linspace(xmin, xmax, 101),
    np.linspace(ymin, ymax, 101)
)

positions_ = np.vstack([xx_.ravel(), yy_.ravel()])
zz_ = kde(positions_).reshape(xx.shape)

# plot
# %%

fig, ax = plt.subplots(1, 1)
cmap_mc = LinearSegmentedColormap.from_list(
    "mc", ["white", "orange","red"]
)

cmap_data = LinearSegmentedColormap.from_list(
    "data", ["white", "lightblue","steelblue"]
)
zz_safe = np.where(zz > 0, zz, 1e-10)
zz_data_safe = np.where(zz_ > 0, zz_, 1e-10)

# choose number of levels
n_levels = 7

# compute levels equally spaced in log space
levels_mc   = np.logspace(np.log10(zz_safe.min()), np.log10(zz_safe.max()), n_levels)
levels_data = np.logspace(np.log10(zz_data_safe.min()), np.log10(zz_data_safe.max()), n_levels)


ax.contourf(xx, yy, zz,  levels=levels_mc, cmap=cmap_mc,   alpha=0.4)
ax.contourf(xx, yy, zz_, levels=levels_data, cmap=cmap_data, alpha=0.4)

ax.contour(xx, yy, zz,  levels=levels_mc, colors="red")
ax.contour(xx, yy, zz_, levels=levels_data, colors="steelblue", linestyles="dashed")

ax.vlines(x=[0.71], ymin=0.71, ymax=1, color="black", linestyle="dotted")
ax.hlines(y=[0.71], xmin=0.71, xmax=1, color="black", linestyle="dotted")
ax.vlines(x=[0.2783], ymin=0.2783, ymax=1, color="black", linestyle="dotted")
ax.hlines(y=[0.2783], xmin=0.2783, xmax=1, color="black", linestyle="dotted")
ax.vlines(x=[0.049], ymin=0.049, ymax=1, color="black", linestyle="dotted")
ax.hlines(y=[0.049], xmin=0.049, xmax=1, color="black", linestyle="dotted")
legend_handles = [
    Patch(facecolor="orange", edgecolor="red", alpha=0.4, label="ggH"),
    Patch(facecolor="lightblue", edgecolor="steelblue", alpha=0.4, label="Data"),
    Line2D([0], [0], color="black", lw=1, linestyle=":", label="b-tag WP"),
]
ax.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=False,
    fontsize=18
)



# define threshold
thr = 0.049

# white mask rectangle
mask = patches.Rectangle(
    (0.0, 0.0),   # bottom-left corner
    1,          # width
    thr,          # height
    facecolor="white",
    edgecolor="none",
    zorder=3      # important: above contourf, below lines if needed
)
ax.add_patch(mask)
mask = patches.Rectangle(
    (0.0, 0.0),   # bottom-left corner
    thr,          # width
    1,          # height
    facecolor="white",
    edgecolor="none",
    zorder=3      # important: above contourf, below lines if needed
)
ax.add_patch(mask)

ax.set_xlabel("jet1 btagDeepFlavB")
ax.set_ylabel("jet2 btagDeepFlavB")



# top marginal
ax_top = inset_axes(
    ax,
    width="100%",
    height="25%",
    loc="upper center",
    bbox_to_anchor=(0, 0.85, 1, 0.25),
    bbox_transform=ax.transAxes,
    borderpad=0
)

# right marginal
ax_right = inset_axes(
    ax,
    width="25%",
    height="100%",
    loc="center right",
    bbox_to_anchor=(0.85, 0, 0.25, 1),
    bbox_transform=ax.transAxes,
    borderpad=0
)

# MC marginals
ax_top.hist(df.jet1_btagDeepFlavB, bins=np.linspace(0, 1, 101), density=True, color='red', alpha=0.4)
ax_top.hist(df_data.jet1_btagDeepFlavB, bins=np.linspace(0, 1, 101), density=True, color='steelblue', alpha=0.4)
ax_top.hist(df.jet1_btagDeepFlavB, bins=np.linspace(0, 1, 101), density=True, color='red', linewidth=3, histtype='step')
ax_top.hist(df_data.jet1_btagDeepFlavB, bins=np.linspace(0, 1, 101), density=True, color='steelblue', linewidth=3, histtype='step')



ax_right.hist(df.jet2_btagDeepFlavB, bins=np.linspace(0, 1, 101), density=True, color='red', alpha=0.4, orientation="horizontal")
ax_right.hist(df_data.jet2_btagDeepFlavB, bins=np.linspace(0, 1, 101), density=True, color='steelblue', alpha=0.4, orientation="horizontal")
ax_right.hist(df.jet2_btagDeepFlavB, bins=np.linspace(0, 1, 101), density=True, color='red', linewidth=3, histtype='step', orientation="horizontal")
ax_right.hist(df_data.jet2_btagDeepFlavB, bins=np.linspace(0, 1, 101), density=True, color='steelblue', linewidth=3, histtype='step', orientation="horizontal")

# Data marginals
ax_top.set_yscale('log')
ax_right.set_xscale('log')

ax_right.set_ylim(ax.get_ylim())
ax_top.set_xlim(ax.get_xlim())
#ax_right.set_xlim(ax.get_xlim())
#ax_top.axis("off")
ax_right.set_xticks([])
ax_right.set_yticks([])
ax_top.set_xticks([])
ax_top.set_yticks([])
#ax_right.axis("off")

plt.tight_layout()
plt.show()
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/btag_categorization_kde.png", bbox_inches='tight')

# %%

fig, ax = plt.subplots(1, 1)
binsbtag = np.linspace(0, 1, 51)
h=ax.hist2d(
    df.jet1_btagDeepFlavB,
    df.jet2_btagDeepFlavB,
    bins=(binsbtag, binsbtag),
    norm=LogNorm(vmin=1, vmax=2*10**3),       
    cmap="jet",
    alpha=0.5,
    
)

ax.set_xlabel("jet1_btagDeepFlavB", fontsize=18)
ax.set_ylabel("jet2_btagDeepFlavB", fontsize=18)
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label("Counts (log scale)", fontsize=14)
ax.vlines(x=[0.71], ymin=0.71, ymax=1, color="black", linestyle="dotted", linewidth=2)
ax.hlines(y=[0.71], xmin=0.71, xmax=1, color="black", linestyle="dotted", linewidth=2)
ax.vlines(x=[0.2783], ymin=0.2783, ymax=1, color="black", linestyle="dotted", linewidth=2)
ax.hlines(y=[0.2783], xmin=0.2783, xmax=1, color="black", linestyle="dotted", linewidth=2)
ax.vlines(x=[0.049], ymin=0.049, ymax=1, color="black", linestyle="dotted", linewidth=2)
ax.hlines(y=[0.049], xmin=0.049, xmax=1, color="black", linestyle="dotted", linewidth=2)
plt.tight_layout()
plt.show()
# %%
df_data_ = df_data.iloc[:len(df)]
fig, ax = plt.subplots(1, 1)
binsbtag = np.linspace(0, 1, 51)
h=ax.hist2d(
    df_data_.jet1_btagDeepFlavB,
    df_data_.jet2_btagDeepFlavB,
    bins=(binsbtag, binsbtag),
    norm=LogNorm(vmin=1, vmax=2*10**3),       
    cmap="jet",
    alpha=0.5
)

ax.set_xlabel("jet1_btagDeepFlavB", fontsize=18)
ax.set_ylabel("jet2_btagDeepFlavB", fontsize=18)
ax.vlines(x=[0.71], ymin=0.71, ymax=1, color="black", linestyle="dotted", linewidth=2)
ax.hlines(y=[0.71], xmin=0.71, xmax=1, color="black", linestyle="dotted", linewidth=2)
ax.vlines(x=[0.2783], ymin=0.2783, ymax=1, color="black", linestyle="dotted", linewidth=2)
ax.hlines(y=[0.2783], xmin=0.2783, xmax=1, color="black", linestyle="dotted", linewidth=2)
ax.vlines(x=[0.049], ymin=0.049, ymax=1, color="black", linestyle="dotted", linewidth=2)
ax.hlines(y=[0.049], xmin=0.049, xmax=1, color="black", linestyle="dotted", linewidth=2)
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label("Counts (log scale)", fontsize=14)
plt.tight_layout()
plt.show()
# %%
import numpy as np
import pandas as pd

wps = [0.049, 0.2783, 0.71, 1.0]  # add 1.0 as upper edge
n = len(wps) - 1

# matrices to store results
s_over_b = np.zeros((n, n))
s_counts = np.zeros((n, n))
b_counts = np.zeros((n, n))
s_over_sqrtb = np.zeros((n, n))

# loop over bins
for i in range(n):
    for j in range(n):
        x_low, x_high = wps[i], wps[i+1]
        y_low, y_high = wps[j], wps[j+1]

        # select signal and background in this bin
        s = df[(df.jet1_btagDeepFlavB >= x_low) & (df.jet1_btagDeepFlavB < x_high) &
               (df.jet2_btagDeepFlavB >= y_low) & (df.jet2_btagDeepFlavB < y_high)].flat_weight.sum()/df.flat_weight.sum()

        b = len(df_data[(df_data.jet1_btagDeepFlavB >= x_low) & (df_data.jet1_btagDeepFlavB < x_high) &
                    (df_data.jet2_btagDeepFlavB >= y_low) & (df_data.jet2_btagDeepFlavB < y_high)])/len(df_data)

        # avoid division by zero
        s_over_b[i, j] = s / b if b > 0 else np.nan
        s_over_sqrtb[i, j] = s / np.sqrt(b) if b > 0 else np.nan
        s_counts[i,j] =s
        b_counts[i,j] =b
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

im1 = axes[0,0].imshow(s_over_b, origin="lower", cmap="Reds", vmin=0)
axes[0,0].set_title("S / B Gain", fontsize=18)
axes[0,0].set_xticks([0,1,2])
axes[0,0].set_yticks([0,1,2])
axes[0,0].set_xticklabels(["loose","medium","tight"])
axes[0,0].set_yticklabels(["loose","medium","tight"])
for i in range(3):
    for j in range(3):
        val = s_over_b[i, j]
        text = f"{val:.2f}" if not np.isnan(val) else "0"
        axes[0,0].text(j, i, text, ha="center", va="center", color="black", fontsize=12)
axes[0,0].set_xlabel("jet1 btagDeepFlavB", fontsize=18)
axes[0,0].set_ylabel("jet2 btagDeepFlavB", fontsize=18)
fig.colorbar(im1, ax=axes[0,0])

im2 = axes[0,1].imshow(s_over_sqrtb, origin="lower", cmap="Blues", vmin=0)
axes[0,1].set_title("S / sqrt(B) Gain", fontsize=18)
axes[0,1].set_xticks([0,1,2])
axes[0,1].set_yticks([0,1,2])
axes[0,1].set_xticklabels(["loose","medium","tight"])
axes[0,1].set_yticklabels(["loose","medium","tight"])
axes[0,1].set_xlabel("jet1 btagDeepFlavB", fontsize=18)
axes[0,1].set_ylabel("jet2 btagDeepFlavB", fontsize=18)
for i in range(3):
    for j in range(3):
        val = s_over_sqrtb[i, j]
        text = f"{val:.2f}" if not np.isnan(val) else "0"
        axes[0,1].text(j, i, text, ha="center", va="center", color="black", fontsize=12)

fig.colorbar(im2, ax=axes[0,1])

im1 = axes[1,0].imshow(s_counts*100, origin="lower", cmap="Greens", vmin=0)
axes[1,0].set_title("S Eff", fontsize=18)
axes[1,0].set_xticks([0,1,2])
axes[1,0].set_yticks([0,1,2])
axes[1,0].set_xticklabels(["loose","medium","tight"])
axes[1,0].set_yticklabels(["loose","medium","tight"])
for i in range(3):
    for j in range(3):
        val = s_counts[i, j]
        text = f"{val*100:.2f}" if not np.isnan(val) else "0"
        axes[1,0].text(j, i, text, ha="center", va="center", color="black", fontsize=12)
axes[1,0].set_xlabel("jet1 btagDeepFlavB", fontsize=18)
axes[1,0].set_ylabel("jet2 btagDeepFlavB", fontsize=18)
fig.colorbar(im1, ax=axes[1,0])



im2 = axes[1,1].imshow(b_counts*100, origin="lower", cmap="Oranges", vmin=0)
axes[1,1].set_title("B Eff", fontsize=18)
axes[1,1].set_xticks([0,1,2])
axes[1,1].set_yticks([0,1,2])
axes[1,1].set_xticklabels(["loose","medium","tight"])
axes[1,1].set_yticklabels(["loose","medium","tight"])
axes[1,1].set_xlabel("jet1 btagDeepFlavB", fontsize=18)
axes[1,1].set_xlabel("jet2 btagDeepFlavB", fontsize=18)
for i in range(3):
    for j in range(3):
        val = b_counts[i, j]
        text = f"{val*100:.2f}" if not np.isnan(val) else "0"
        axes[1,1].text(j, i, text, ha="center", va="center", color="black", fontsize=12)
axes[1,1].set_xlabel("jet1 btagDeepFlavB", fontsize=18)


axes[1,1].set_ylabel("jet2 btagDeepFlavB", fontsize=18)
fig.colorbar(im2, ax=axes[1,1])
plt.tight_layout()
plt.show()

# %%

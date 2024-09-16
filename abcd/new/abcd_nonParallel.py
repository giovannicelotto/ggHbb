# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")

# %%
def getData():
    #nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A"
    #flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt20To30"
    fileNames = glob.glob(flatPathCommon+"/*/*.parquet", recursive=True)[:500]

    df = pd.read_parquet(fileNames, columns=['dijet_mass', 'jet2_btagDeepFlavB', 'dijet_twist', 'jet1_pt', 'jet2_pt', 'dijet_pt'])
    return df
# %%

df = getData()
df = df[(df.jet1_pt>20) & (df.jet2_pt>20) & (df.dijet_pt>100)]

# variable x1 and x2

x1_threshold = 1
x2_threshold = 0.1
x1 = 'dijet_twist'
x2 = 'jet2_btagDeepFlavB'
xx = 'dijet_mass'
# %%

bins = np.linspace(40, 200, 21)
x = (bins[:-1] + bins[1:])/2
fig, ax = plt.subplots(1, 1)
counts = np.histogram(df.dijet_mass, bins=bins)[0]
errors = np.sqrt(counts)
ax.errorbar(x, counts, yerr=errors, linestyle='none', color='black', markersize=2, marker='o')

# %%
regions = {
    'A':np.zeros(len(bins)-1),
    'B':np.zeros(len(bins)-1),
    'D':np.zeros(len(bins)-1),
    'C':np.zeros(len(bins)-1),
}

# %%


        
regions['A'] = regions['A'] + np.histogram(df[(df[x1]< x1_threshold) & (df[x2]>=  x2_threshold)][xx], bins=bins)[0]
regions['B'] = regions['B'] + np.histogram(df[(df[x1]>=x1_threshold) & (df[x2]>=  x2_threshold)][xx], bins=bins)[0]
regions['C'] = regions['C'] + np.histogram(df[(df[x1]< x1_threshold) & (df[x2] <  x2_threshold)][xx], bins=bins)[0]
regions['D'] = regions['D'] + np.histogram(df[(df[x1]>=x1_threshold) & (df[x2] <  x2_threshold)][xx], bins=bins)[0]

# %%




fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 10))
x=(bins[1:]+bins[:-1])/2
ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"], histtype=u'step', label='Region A')
ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"], histtype=u'step', label='Region B')
ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'], histtype=u'step', label='Region C')
ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'], histtype=u'step', label='Region D')

ax[0,1].hist(bins[:-1], bins=bins, weights=regions['A']*regions['D']/(regions['C']+1e-6), histtype=u'step', label=r'$A\times D / C$ ')
#ax[0,2].hist(bins[:-1], bins=bins, weights=regions["B"]*regions['C']/(regions['A']*regions['D']), histtype=u'step', label=r'B$\times$C/A$\times$D')


ax[0,0].set_title("%s < %.1f, %s >= %.1f"%(x1, x1_threshold, x2, x2_threshold), fontsize=14)
ax[0,1].set_title("%s >= %.1f, %s >= %.1f"%(x1, x1_threshold, x2, x2_threshold), fontsize=14)
ax[1,0].set_title("%s < %.1f, %s < %.1f"%(x1, x1_threshold, x2, x2_threshold), fontsize=14)
ax[1,1].set_title("%s >= %.1f, %s < %.1f"%(x1, x1_threshold, x2, x2_threshold), fontsize=14)
for idx, axx in enumerate(ax.ravel()):
    axx.set_xlim(bins[0], bins[-1])
    axx.set_xlabel("Dijet Mass [GeV]")
    axx.legend(fontsize=18, loc='upper right')
#fig.savefig("", bbox_inches='tight')
#print("Saving in ", "/t3home/gcelotto/ggHbb/abcd/output/abcd.png")

# %%
fig, ax = plt.subplots(1, 1)
b_err = np.sqrt(regions['B'])
adc_err = np.sqrt(regions['A']*regions['D']/regions['C'])
ax.errorbar(x, regions['B']-regions['A']*regions['D']/regions['C'], yerr=np.sqrt(b_err**2 + adc_err**2) , marker='o', color='black', linestyle='none')
# %%
for letter in ['A', 'B', 'C', 'D']:
    print(np.sum(regions[letter]))
# %%
    


import seaborn as sns
# Create a scatter plot to visualize correlation
plt.figure(figsize=(10, 8))
plt.scatter(df[x1], df[x2], alpha=0.5, c='blue', edgecolor='k', s=20)
plt.title(f"Scatter Plot of {x1} vs {x2}", fontsize=16)
plt.xlabel(x1, fontsize=14)
plt.ylabel(x2, fontsize=14)
plt.grid(True)
plt.show()

# Create a 2D histogram to visualize correlation density
plt.figure(figsize=(10, 8))
plt.hist2d(df[x1], df[x2], bins=50, cmap='viridis')
plt.colorbar(label='Counts')
plt.title(f"2D Histogram of {x1} vs {x2}", fontsize=16)
plt.xlabel(x1, fontsize=14)
plt.ylabel(x2, fontsize=14)
plt.grid(True)
plt.show()

# Calculate the Pearson correlation coefficient
correlation = df[[x1, x2]].corr().iloc[0, 1]
print(f"Pearson correlation coefficient between {x1} and {x2}: {correlation:.4f}")

# Create a correlation heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(df[[x1, x2]].corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# %%

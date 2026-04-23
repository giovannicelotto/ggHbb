# %%
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
# %%
fig, ax = plt.subplots(1, 1, figsize=(10,4))
sf = [0.812142, 0.772752, 0.770822]
sf_err = [0.0979795, 0.073858, 0.113264]
ax.errorbar([0, 1, 2], sf, yerr=sf_err, fmt='o', color='black', ecolor='red', capsize=5)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['0.85 < NN < 0.875', '0.875 < NN < 0.925', 'NN > 0.925'], rotation=30)
ax.set_ylim(0.6, 1.)
ax.set_ylabel("Scale Factor")
# %%

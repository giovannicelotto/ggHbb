import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from mpl_toolkits.axes_grid1 import make_axes_locatable
hep.style.use("CMS")
path = "/t3home/gcelotto/ggHbb/outputs/wrongJetsMassCriterion.npy"

pt, mass = np.load(path)[:,0], np.load(path)[:,1]
x_bins, y_bins = np.linspace(0, 200, 30), np.linspace(0, 200, 30)
fig, ax_main = plt.subplots(figsize=(8, 8))
divider = make_axes_locatable(ax_main)
ax_top = divider.append_axes("top", 1.2, pad=0.2, sharex=ax_main)
ax_right = divider.append_axes("right", 1.2, pad=0.2, sharey=ax_main)

# Plot the 2D histogram in the main axes
hist, x_edges, y_edges = np.histogram2d(x=pt, y=mass, bins=[x_bins, y_bins])
ax_main.imshow(hist.T, origin='lower', extent=(x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()), aspect='auto', cmap='Blues')
ax_main.set_xlabel("Dijet Pt [GeV]")
ax_main.set_ylabel("Dijet Mass [GeV]")

# Plot the marginalized histogram on top
ax_top.hist(pt, bins=x_bins, color='lightblue', edgecolor='black')
ax_top.set_xlim(ax_main.get_xlim())
ax_top.set_yticks([])
ax_top.xaxis.tick_top()

# Plot the marginalized histogram on the right
ax_right.hist(mass, bins=y_bins, color='lightblue', edgecolor='black', orientation='horizontal')#lightcoral
ax_right.set_ylim(ax_main.get_ylim())
ax_right.set_xticks([])
ax_right.yaxis.tick_right()

outName = "/t3home/gcelotto/ggHbb/outputs/plots/wrongMass.png"
print("Saving in ", outName)
fig.savefig(outName, bbox_inches='tight')
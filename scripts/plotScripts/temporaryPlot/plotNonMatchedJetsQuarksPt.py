import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from mpl_toolkits.axes_grid1 import make_axes_locatable
hep.style.use("CMS")

def plotPtEtaPdg(path, tag):
    a=np.load(path)
    print(a.shape)
    pt1, eta1, pt2, eta2 = np.load(path)[:,0], np.load(path)[:,1], np.load(path)[:,2], np.load(path)[:,3]#, np.load(path)[:,2]

    x_bins, y_bins = np.linspace(-1, 1, 30), np.linspace(-1, 1, 30)
    fig, ax_main = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(wspace=0.3)
    ax_main[0].text(x=0.05, y=0.9, s="Leading Jet vs Quark", transform=ax_main[0].transAxes, fontsize=16)
    ax_main[1].text(x=0.05, y=0.9, s="Subleading Jet vs Quark", transform=ax_main[1].transAxes, fontsize=16)
    
    divider = make_axes_locatable(ax_main[0])
    ax_top = divider.append_axes("top", 1.2, pad=0.2, sharex=ax_main[0])
    ax_right = divider.append_axes("right", 1.2, pad=0.2, sharey=ax_main[0])

    # Plot the 2D histogram in the main axes
    hist, x_edges, y_edges = np.histogram2d(x=pt1, y=eta1, bins=[x_bins, y_bins])
    ax_main[0].imshow(hist.T, origin='lower', extent=(x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()), aspect='auto', cmap='Blues')
    ax_main[0].set_xlabel("$(p_\mathrm{T\,jet}^\mathrm{reco} - p_\mathrm{T quark}^\mathrm{gen})/p_\mathrm{T quark}^\mathrm{gen}$")
    ax_main[0].set_ylabel("$(\eta_\mathrm{jet}^\mathrm{reco} - \eta_\mathrm{quark}^\mathrm{gen})/\eta_\mathrm{quark}^\mathrm{gen}$")

    # Plot the marginalized histogram on top
    ax_top.hist(pt1, bins=x_bins, color='lightblue', edgecolor='black')
    ax_top.set_xlim(ax_main[0].get_xlim())
    ax_top.set_yticks([])
    ax_top.xaxis.tick_top()

    # Plot the marginalized histogram on the right
    ax_right.hist(eta1, bins=y_bins, color='lightblue', edgecolor='black', orientation='horizontal')#lightcoral
    ax_right.set_ylim(ax_main[0].get_ylim())
    ax_right.set_xticks([])
    ax_right.yaxis.tick_right()





    # Second main axis
    divider = make_axes_locatable(ax_main[1])
    ax_top = divider.append_axes("top", 1.2, pad=0.2, sharex=ax_main[1])
    ax_right = divider.append_axes("right", 1.2, pad=0.2, sharey=ax_main[1])

    # Plot the 2D histogram in the main axes
    hist, x_edges, y_edges = np.histogram2d(x=pt2, y=eta2, bins=[x_bins, y_bins])
    ax_main[1].imshow(hist.T, origin='lower', extent=(x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()), aspect='auto', cmap='Blues')
    ax_main[1].set_xlabel("$(p_\mathrm{T\,jet}^\mathrm{reco} - p_\mathrm{T quark}^\mathrm{gen})/p_\mathrm{T quark}^\mathrm{gen}$")
    ax_main[1].set_ylabel("$(\eta_\mathrm{jet}^\mathrm{reco} - \eta_\mathrm{quark}^\mathrm{gen})/\eta_\mathrm{quark}^\mathrm{gen}$")

    # Plot the marginalized histogram on top
    ax_top.hist(pt2, bins=x_bins, color='lightblue', edgecolor='black')
    ax_top.set_xlim(ax_main[1].get_xlim())
    ax_top.set_yticks([])
    ax_top.xaxis.tick_top()

    # Plot the marginalized histogram on the right
    ax_right.hist(eta2, bins=y_bins, color='lightblue', edgecolor='black', orientation='horizontal')#lightcoral
    ax_right.set_ylim(ax_main[1].get_ylim())
    ax_right.set_xticks([])
    ax_right.yaxis.tick_right()

    #ax_main[1].hist(pdgId, bins=1000)

    fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/%sJetsQuarksPt.png"%tag)
path = "/t3home/gcelotto/ggHbb/outputs/nonMatchedQuarksPt.npy"
tag = 'nonMatched'
plotPtEtaPdg(path=path, tag=tag)
#path = "/t3home/gcelotto/ggHbb/outputs/matchedQuarksPt.npy"
#tag = 'matched'
#plotPtEtaPdg(path=path, tag=tag)


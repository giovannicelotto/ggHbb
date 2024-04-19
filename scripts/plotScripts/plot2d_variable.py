import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplhep as hep
hep.style.use("CMS")
def main():
    flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    flatFileNames = glob.glob(flatPath+"/GluGluHToBB/**/*.parquet", recursive=True)[:100]
    df = pd.read_parquet(flatFileNames, columns=['jet1_btagDeepFlavB', 'jet1_pt','dijet_twist'])
    flatFileNames = glob.glob(flatPath+"/Data1A/**/*.parquet", recursive=True)[:10]
    df_data = pd.read_parquet(flatFileNames, columns=['jet1_btagDeepFlavB', 'jet1_pt','dijet_twist'])

    
    x_bins, y_bins = np.linspace(0, 1, 20), np.linspace(0, np.pi/2, 20)
    fig, ax_main = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", 1.2, pad=0.2, sharex=ax_main)
    ax_right = divider.append_axes("right", 1.2, pad=0.2, sharey=ax_main)

    # Plot the 2D histogram in the main axes
    hist, x_edges, y_edges = np.histogram2d(x=df.jet1_btagDeepFlavB, y=df.dijet_twist, bins=[x_bins, y_bins])
    ax_main.imshow(hist.T, origin='lower', extent=(x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()), aspect='auto', cmap='Blues')
    ax_main.set_xlabel("Jet1 btag")
    ax_main.set_ylabel("Dijet twist")

    # Plot the marginalized histogram on top
    ax_top.hist(df.jet1_btagDeepFlavB, bins=x_bins, color='lightblue', edgecolor='black')
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_yticks([])
    ax_top.xaxis.tick_top()

    # Plot the marginalized histogram on the right
    ax_right.hist(df.dijet_twist, bins=y_bins, color='lightblue', edgecolor='black', orientation='horizontal')#lightcoral
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_xticks([])
    ax_right.yaxis.tick_right()

    #outName = "/t3home/gcelotto/ggHbb/outputs/plots/wrongMass.png"
    #print("Saving in ", outName)
    #fig.savefig(outName, bbox_inches='tight')
    #fig.savefig("")
    #fig = corner.corner(df, show_titles=True,)
    #fig = corner.corner(df_data, show_titles=True,fig=fig)
#
    #fig=corner.corner(df,labels=df.columns,levels=(0.5,0.9, 0.99),  color='tab:blue', scale_hist=True,plot_density=True,bins=30)
    #corner.corner(df_data[:len(df)],labels=df.columns,levels=(0.5,0.9, 0.99), fig=fig, color='tab:orange', scale_hist=True,bins=30,plot_density=True)
    #
    outName = "/t3home/gcelotto/ggHbb/outputs/plots/hist2d_jet1btag_dijettwist.png"
    fig.savefig(outName, bbox_inches='tight')
    print("Saving in ", outName)
    return

if __name__ =="__main__":
    main()
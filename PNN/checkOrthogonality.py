from scipy.stats import ks_2samp, chisquare, chi2
import numpy as np
import mplhep as hep
hep.style.use("CMS")
import matplotlib.pyplot as plt
def checkOrthogonality(df, featureToPlot, mask1, mask2, label_mask1, label_mask2, label_toPlot, bins, ax=None, axLegend=False, outName=None ):
    nReal = 0
    """
    Plot normalized histograms and their ratio with error bars for two data masks.

    Parameters:
    - df: DataFrame, the input dataset
    - featureToPlot: str, column name of the feature to plot
    - mask1, mask2: Boolean masks, the two masks to compare
    - label_mask1, label_mask2: str, labels for the two masks
    - label_toPlot: str, label for the x-axis of the plot
    - bins: array-like, bin edges for the histogram
    - outName: str, path of the folder and name where to save the plot (default None --> plot not saved)
    """
    eps=1e-10
    hNN_Tight = np.histogram(df[mask1][featureToPlot], bins=bins)[0]
    hNN_Medium = np.histogram(df[mask2][featureToPlot], bins=bins)[0]
    norm_Tight = np.sum(hNN_Tight)
    norm_Medium = np.sum(hNN_Medium)
    err_Tight = np.sqrt(hNN_Tight)
    err_Medium = np.sqrt(hNN_Medium)

    ks_stat, ks_p_value = ks_2samp(df[mask1][featureToPlot], df[mask2][featureToPlot])
    print(f"KS Test: statistic = {ks_stat:.4f}, p-value = {ks_p_value:.4f}")


    # Normalize counts and uncertainties
    hNN_Tight = hNN_Tight / norm_Tight
    hNN_Medium = hNN_Medium / norm_Medium
    err_Tight = err_Tight / norm_Tight
    err_Medium = err_Medium / norm_Medium

    observed = np.maximum(hNN_Tight, 1e-10)
    expected = np.maximum(hNN_Medium, 1e-10)
    chi2_stat =  sum((observed-expected)**2/(err_Tight**2+err_Medium**2+eps))
    p_value = chi2.sf(chi2_stat, len(observed)-2)
    print(f"Chi-squared / ndof = {chi2_stat:.4f} / {len(observed)-2:d}")
    
    if ax==None:
        fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
        ax[0].hist(bins[:-1], bins=bins, weights=hNN_Tight, histtype='step', linewidth=2, label=label_mask1, color='black')[0]
        ax[0].hist(bins[:-1], bins=bins, weights=hNN_Medium, histtype='step',  linewidth=2, linestyle='dashed', label=label_mask2, color='red')[0]
        ax[0].legend()
        ax[1].errorbar((bins[1:]+bins[:-1])/2, y=hNN_Tight/(hNN_Medium), yerr=hNN_Tight/hNN_Medium*np.sqrt(err_Tight**2/hNN_Tight**2 + err_Medium**2/hNN_Medium**2), linestyle='none', color='black', marker='o')
        ax[1].hlines(y=1, xmin=bins[0], xmax=bins[-1], color='red')
        ax[0].set_ylabel("Normalized Counts")
        #ax[0].text(x=0.25, y=0.5, s="KS Test: statistic = %.3f"%(ks_stat), transform=ax[0].transAxes)
        ax[0].text(x=0.25, y=0.45, s="KS Test: pvalue = %.2f"%(ks_p_value), transform=ax[0].transAxes)
        ax[0].text(x=0.25, y=0.4, s="$\chi^2$ / ndof = %.1f / %d (pval = %.2f)"%(chi2_stat, len(observed)-2, p_value), transform=ax[0].transAxes)
        ax[1].set_ylabel("Ratio")
        ax[1].set_xlabel(label_toPlot)
        hep.cms.label(ax=ax[0], lumi = np.round(0.774*nReal/1017, 3))
        if outName is not None:
            fig.savefig(outName)
    else:
        ax.errorbar((bins[1:]+bins[:-1])/2, y=hNN_Tight/(hNN_Medium+eps), yerr=hNN_Tight/(hNN_Medium+eps)*np.sqrt(err_Tight**2/(hNN_Tight+eps)**2 + err_Medium**2/(hNN_Medium+eps)**2), linestyle='none', color='black', marker='o', label=label_mask1)
        ax.hlines(y=1, xmin=bins[0], xmax=bins[-1], color='red', label=label_mask2)
        ax.set_xlabel(label_toPlot)
        if axLegend:
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
            ax.legend()
        ax.text(x=0.02, y=0.85, s="KS Test: pvalue = %.2f"%(ks_p_value), transform=ax.transAxes, fontsize=14)
        ax.text(x=0.02, y=0.75, s="$\chi^2$ / ndof = %.1f / %d (pval = %.2f)"%(chi2_stat, len(observed)-2, p_value), transform=ax.transAxes, fontsize=14)
        return ks_p_value, p_value, chi2_stat
        



def checkOrthogonalityInMassBins(df, featureToPlot, mask1, mask2, label_mask1, label_mask2, label_toPlot, bins, mass_bins, mass_feature='dijet_mass', figsize=(15, 15), outName=None):
    # Create the figure grid based on the number of mass bins
    n_mass_bins = len(mass_bins) - 1
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_mass_bins + n_cols - 1) // n_cols  # Calculate rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True, sharex='row', sharey='row')
    axes = axes.flatten()
    ks_p_value_list=[]
    p_value_list=[]
    chi2_values_list=[]
    # Loop through dijet_mass bins and plot for each
    for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        # Define mass mask
        mask_mass = (df[mass_feature] >= low) & (df[mass_feature] < high)
        # Apply mass mask to both comparison masks
        combined_mask1 = mask1 & mask_mass
        combined_mask2 = mask2 & mask_mass

        # Check if there is any data in this bin, skip if empty
        if not combined_mask1.any() or not combined_mask2.any():
            axes[i].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', fontsize=12)
            axes[i].set_title(f"{low:.1f} ≤ {mass_feature} < {high:.1f}")
            axes[i].axis('off')
            continue

        if i % n_cols == 0:  
            axes[i].set_ylabel("Ratio")
        

        # Call the checkOrthogonality function for this mass bin
        
        #axLegend=True if i==0 else False
        axLegend=False
        ks_p_value, p_value, chi2value = checkOrthogonality(
            df=df, featureToPlot=featureToPlot, mask1=combined_mask1, mask2=combined_mask2,
            label_mask1=label_mask1, label_mask2=label_mask2, label_toPlot=label_toPlot, bins=bins, ax=axes[i],
            axLegend=axLegend
        )
        ks_p_value_list.append(ks_p_value)
        p_value_list.append(p_value)
        chi2_values_list.append(chi2value)
        axes[i].set_title(f"{low:.1f} ≤ {mass_feature} < {high:.1f}")

    # Turn off unused subplots
    for ax in axes[n_mass_bins:]:
        ax.axis('off')

    # Set common labels
    #fig.text(0.5, 0.04, label_toPlot, ha='center')
    #fig.text(0.04, 0.5, "Normalized Counts / Ratio", va='center', rotation='vertical')
    plt.show()

    if outName is not None:
            fig.savefig(outName)
    return ks_p_value_list, p_value_list, chi2_values_list
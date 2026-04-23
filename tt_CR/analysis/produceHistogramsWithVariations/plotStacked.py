import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
hep.style.use("CMS")
import os
from binning_per_variable import plot_vars
def plot_data_mc_stack_ratio(dfData, dfMC, var, bins, xlabel, lumi):
    fig, (ax, rax) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    print("Plotting variable %s with %d bins"%(var, len(bins)-1))
    # -----------------
    # Main (Data + MC)
    # -----------------
    mask_data = dfData.is_ttbar_CR == 1
    data_vals = dfData.loc[mask_data, var]
    data_counts, _ = np.histogram(data_vals, bins=bins)

    ax.errorbar(
        (bins[1:] + bins[:-1]) / 2,
        data_counts,
        yerr=np.sqrt(data_counts),
        marker="o",
        linestyle="none",
        color="black",
        label="Data",
    )
    print("Data counts:", data_counts)

    mask_mc = dfMC.is_ttbar_CR == 1
    print("MC processes available:", dfMC.process.unique())
    processes = list(np.unique(dfMC.process))
    print("MC processes to consider:", processes)
    mc_arrays = []
    mc_weights = []
    mc_processes = []

    for proc in processes:
        print("Processing MC process:", proc)
        if proc == "ggH(bb)":
            continue
        sel = (mask_mc) & (dfMC.process == proc)
        mc_arrays.append(dfMC.loc[sel, var])
        mc_weights.append(dfMC.loc[sel, "weight"])
        mc_processes.append(proc)
    print("MC processes to plot:", mc_processes)
    mc_counts = np.histogram(
        np.concatenate(mc_arrays),
        bins=bins,
        weights=np.concatenate(mc_weights),
    )[0]
    print("MC counts (unstacked):", mc_counts)
    ax.hist(
        mc_arrays,
        bins=bins,
        weights=mc_weights,
        stacked=True,
        label=processes,
    )
    print("MC counts (stacked):", mc_counts)
    if "PNN" in var:
        var_ggH = "PNN"
    else:
        var_ggH = var
    ax.hist(
        dfMC[dfMC.process=="ggH(bb)"][var_ggH],
        bins=bins,
        weights=dfMC.weight[dfMC.process=="ggH(bb)"],
        histtype="step",
        color='red',
        linewidth=3,
        label="ggH(bb)",
    )
    ax.set_ylabel("Events")
    ax.set_ylim(0, 1.4 * max(data_counts.max(), mc_counts.max()))
    ax.legend(loc='best')

    # -----------------
    # Ratio
    # -----------------
    ratio = np.divide(
        data_counts,
        mc_counts,
        out=np.zeros_like(data_counts, dtype=float),
        where=mc_counts > 0,
    )

    ratio_err = np.divide(
        np.sqrt(data_counts),
        mc_counts,
        out=np.zeros_like(data_counts, dtype=float),
        where=mc_counts > 0,
    )
    print("Ratio:", ratio)
    centers = (bins[1:] + bins[:-1]) / 2

    rax.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        marker="o",
        linestyle="none",
        color="black",
    )
    mc_sumw2 = np.histogram(
    np.concatenate(mc_arrays),
    bins=bins,
    weights=np.concatenate(mc_weights) ** 2,
    )[0]

    mc_unc = np.sqrt(mc_sumw2)
    ratio_mc_unc = np.divide(   mc_unc, mc_counts,
                                out=np.zeros_like(mc_unc, dtype=float), where=mc_counts > 0)
    y_low = 1 - ratio_mc_unc
    y_high = 1 + ratio_mc_unc

    # Extend arrays to match bin edges
    y_low = np.r_[y_low, y_low[-1]]
    y_high = np.r_[y_high, y_high[-1]]

    rax.fill_between(
                        bins, y_low, y_high,
                        step="post",color="gray",alpha=0.4,linewidth=0,
    )
    rax.axhline(1.0, linestyle="--", linewidth=1)
    rax.set_ylim(0.5, 1.5)
    ax.set_xlim(bins[0], bins[-1])
    rax.set_ylabel("Data / MC")
    rax.set_xlabel(xlabel)

    # Align y-labels
    fig.align_ylabels([ax, rax])
    hep.cms.label(lumi=np.round(lumi,2), ax=ax)

    return fig, (ax, rax), mc_arrays, mc_weights




def plot_all_variables(dfData, dfMC, lumi, config, folder):
    plot_vars["nJets_20"] ={
        "bins": np.arange(10)-0.5,
        "xlabel": "nJets_20"}
    for var, cfg in plot_vars.items():
        if var in config["columns"] or var == "dijet_dEta":
            pass
        else:
            #print("[WARNING] Variable %s not found in dataframe columns, skipping..."%var)
            continue
        print(var," plotting...")
        
        fig, (ax, rax), mc_arrays, mc_weights = plot_data_mc_stack_ratio(
            dfData,
            dfMC,
            var=var,
            bins=cfg["bins"],
            xlabel=cfg["xlabel"],
            lumi=lumi,
        )
        print("Creating folder...")
        os.makedirs(folder, exist_ok=True)
        fig.savefig(folder+"/%s.png"%(var), bbox_inches="tight")
        plt.close('all')
        del fig, ax, rax


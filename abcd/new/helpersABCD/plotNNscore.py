import matplotlib.pyplot as plt
import numpy as np
def plotNNscore(dfsData):
    fig, ax = plt.subplots(2, 2, figsize=(8, 10))
    nn_bins = np.linspace(0, 1, 71)
    for idx, df in enumerate(dfsData):
        # Compute the histogram for the first dataset (reference) and normalize
        c0 = np.histogram((df.PNN1), bins=nn_bins)[0]
        err_c0 = np.sqrt(c0)/np.sum(c0)
        c0 = c0/np.sum(c0)  # Normalize the histogram
        ax[0,0].hist(nn_bins[:-1], bins=nn_bins, weights=c0, histtype='step', linewidth=len(dfsData)-idx)

        c1 = np.histogram((df.PNN1), bins=nn_bins)[0]
        err_c1 = np.sqrt(c1)/np.sum(c1)
        c1 = c1/np.sum(c1)  # Normalize the histogram
        residuals = c1 - c0
        ax[1,0].errorbar(nn_bins[:-1], residuals,yerr=np.sqrt(err_c0**2+err_c1**2), label=f"Dataset {idx+1}", linestyle='none', marker='o')

        c0 = np.histogram((df.PNN2), bins=nn_bins)[0]
        err_c0 = np.sqrt(c0)/np.sum(c0)
        c0 = c0/np.sum(c0)  # Normalize the histogram
        ax[0,1].hist(nn_bins[:-1], bins=nn_bins, weights=c0, histtype='step', linewidth=len(dfsData)-idx)


        c1 = np.histogram((df.PNN2), bins=nn_bins)[0]
        err_c1 = np.sqrt(c1)/np.sum(c1)
        c1 = c1/np.sum(c1)  # Normalize the histogram
        residuals = c1 - c0
        ax[1,1].errorbar(nn_bins[:-1], residuals,yerr=np.sqrt(err_c0**2+err_c1**2), label=f"Dataset {idx+1}", linestyle='none', marker='o')
    plt.tight_layout()
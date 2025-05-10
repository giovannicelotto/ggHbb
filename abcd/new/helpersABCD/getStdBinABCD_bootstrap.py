import numpy as np
from scipy.stats import pearsonr
import os

def bootstrap_B_corr(df_mass_bin, dfMC_mass_bin, m_fit, q_fit, t1, t2, n_bootstrap=10000):
    # Initialize arrays to store bootstrap values
    B_corr_bootstrap = []
    
    # Get original data
    PNN1 = df_mass_bin['PNN1'].to_numpy()
    PNN2 = df_mass_bin['PNN2'].to_numpy()

    PNN1_MC = dfMC_mass_bin['PNN1'].to_numpy()
    PNN2_MC = dfMC_mass_bin['PNN2'].to_numpy()
    
    mA_MC = (PNN1_MC < t1) & (PNN2_MC > t2)
    mB_MC = (PNN1_MC > t1) & (PNN2_MC > t2)
    mC_MC = (PNN1_MC < t1) & (PNN2_MC < t2)
    mD_MC = (PNN1_MC > t1) & (PNN2_MC < t2)

    nA_MC = dfMC_mass_bin[mA_MC].weight.sum()
    nB_MC = dfMC_mass_bin[mB_MC].weight.sum()
    nC_MC = dfMC_mass_bin[mC_MC].weight.sum()
    nD_MC = dfMC_mass_bin[mD_MC].weight.sum()
    print(nA_MC, nB_MC, nC_MC, nD_MC)

    # Define the correction function gamma (can be your linear model)
    def gamma(x1, x2, m_fit, q_fit):
        pearson_coef, _ = pearsonr(x1, x2)
        return 1 / (q_fit + m_fit * pearson_coef)  # Example formula

    # Bootstrap resampling loop
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(len(df_mass_bin), size=len(df_mass_bin), replace=True)
        
        pnn1_sample = PNN1[idx]
        pnn2_sample = PNN2[idx]
        
        # Calculate mA, mB, mC, mD
        mA = (pnn1_sample < t1) & (pnn2_sample > t2)
        mB = (pnn1_sample > t1) & (pnn2_sample > t2)
        mC = (pnn1_sample < t1) & (pnn2_sample < t2)
        mD = (pnn1_sample > t1) & (pnn2_sample < t2)

        
        

        nA = mA.sum() - nA_MC
        nD = mD.sum() - nD_MC
        nC = mC.sum() - nC_MC
        
        # If nC > 0, compute the bootstrap corrected prediction B_corr
        if nC > 0:
            gamma_value = gamma(pnn1_sample, pnn2_sample, m_fit, q_fit)
            B_corr_bootstrap.append((nA * nD / nC) * gamma_value)
    
    # Calculate the standard deviation (uncertainty) from the bootstrap distribution
    B_corr_mean = np.mean(B_corr_bootstrap)
    std_B_corr = np.std(B_corr_bootstrap)
    
    return B_corr_mean, std_B_corr

def getStdAllBins(dfData, dfMC, xx, bins, m_fit, q_fit, t1, t2, save, path):
    
    if save==False:
        if os.path.exists(path):
            B_corr_stds = np.load(path)
            return B_corr_stds
        else: 
            "File not found. Producing toys..."
    B_corr_means = []
    B_corr_stds = []

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]


        # Slice data for current dijet mass bin
        df_bin = dfData[(dfData[xx] > low) & (dfData[xx] <= high)]
        dfMC_bin = dfMC[(dfMC[xx] > low) & (dfMC[xx] <= high)]

        if len(df_bin) == 0:
            B_corr_means.append(np.nan)
            B_corr_stds.append(np.nan)
            continue
        
        mean, std = bootstrap_B_corr(df_bin, dfMC_bin, m_fit, q_fit, t1, t2, n_bootstrap=300)
        B_corr_means.append(mean)
        B_corr_stds.append(std)

    #np.save("/t3home/gcelotto/ggHbb/abcd/new/output/Bin%d_%d_std.npy"%(), B_corr_stds)

    print("%d - %d : "%(low, high), std)
    print("saving path ", path)
    np.save(path, B_corr_stds)
    return B_corr_stds
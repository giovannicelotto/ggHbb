from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def runPearson(dfsData, bins, withErrors=True, num_bootstrap = 100, fraction = 1 / 5  ):
    pearson_data_values = []
    df = pd.concat(dfsData)
    # Parameters
    # Number of bootstrap resamples (Number of toys)
    # Use 1/5 of total events per bin

    pearson_data_values = []
    bootstrap_errors = []
    confidence_intervals = []

    for b_low, b_high in zip(bins[:-1], bins[1:]):
        m = (df.dijet_mass > b_low) & (df.dijet_mass < b_high)
        x = np.array(df.PNN1[m], dtype=np.float64)
        y = np.array(df.PNN2[m], dtype=np.float64)

        # Compute Pearson correlation for the full bin
        pearson_coef = pearsonr(x, y)[0]
        pearson_data_values.append(pearson_coef)
        if withErrors:
            # Bootstrap resampling
            n_samples = int(len(x) * fraction)  # Use 1/5 of data in each resample
            boot_corrs = []

            for _ in range(num_bootstrap):
                idx = np.random.choice(len(x), n_samples, replace=True)
                boot_corrs.append(pearsonr(x[idx], y[idx])[0])

            # Compute standard error and confidence interval
            se = np.std(boot_corrs)  
            ci_lower, ci_upper = np.percentile(boot_corrs, [2.5, 97.5])  
            plt.hist(np.clip(boot_corrs, -0.02, 0.02), bins=np.linspace(-0.02, 0.02, 101))
            #plt.yscale('log')
            plt.show()
            plt.close()

            bootstrap_errors.append(se)
            confidence_intervals.append((ci_lower, ci_upper))

        # Print results for each bin
            print(f"mjj in ({b_low:.1f}, {b_high:.1f}): r = {pearson_coef:.5f}, SE = {se:.5f}, 95% CI = ({ci_lower:.5f}, {ci_upper:.5f})")
        else:
            print(f"mjj in ({b_low:.1f}, {b_high:.1f}): r = {pearson_coef:.5f}")

# Convert results to numpy arrays
    pearson_data_values = np.array(pearson_data_values)
    bootstrap_errors = np.array(bootstrap_errors) if withErrors else None
    confidence_intervals = np.array(confidence_intervals) if withErrors else None
    return pearson_data_values, bootstrap_errors, confidence_intervals



def makePlotsPearson(pulls_QCD_SR, err_pulls_QCD_SR, pearson_data_values, bins, bootstrap_errors):
    bootstrap_errors=np.zeros(len(pulls_QCD_SR)) if bootstrap_errors is None else bootstrap_errors
    fig, ax = plt.subplots(1, 1)
    maskBlind = pulls_QCD_SR>0
    condition1 = pearson_data_values[maskBlind] > 0
    condition2 = pulls_QCD_SR[maskBlind]-1 > 0
    yTrue = pulls_QCD_SR[maskBlind]-1
    ax.bar(((bins[:-1]+bins[1:])/2)[maskBlind], np.where(condition1, yTrue, -yTrue), width=1, label='Sign of pearson')
    ax.bar(((bins[:-1]+bins[1:])/2)[maskBlind], np.where(condition2, yTrue*0.5, -yTrue*0.5), width=1, label='Sign of Pull')
    ax.legend()
    plt.show()
    plt.close()



    fig, ax = plt.subplots(3, 1)
    x=(bins[1:] + bins[:-1])/2
    ax[0].errorbar(x[pulls_QCD_SR>0], pearson_data_values[pulls_QCD_SR>0], bootstrap_errors[pulls_QCD_SR>0], linestyle='none', marker='o')
    ax[0].set_xlabel("Dijet Mass")
    ax[0].set_ylabel("Pearson")
    ax[1].errorbar(x[pulls_QCD_SR>0], pulls_QCD_SR[pulls_QCD_SR>0], err_pulls_QCD_SR[pulls_QCD_SR>0], linestyle='none', marker='o')
    ax[1].set_xlabel("Dijet Mass")
    ax[1].set_ylabel("Ratio")
    ax[2].errorbar(pearson_data_values[pulls_QCD_SR>0], pulls_QCD_SR[pulls_QCD_SR>0], yerr=err_pulls_QCD_SR[pulls_QCD_SR>0], xerr=bootstrap_errors[pulls_QCD_SR>0], linestyle='none', marker='o')
    ax[2].set_xlabel("Pearson")
    ax[2].set_ylabel("Ratio")
    plt.show()
    plt.close()




    return maskBlind

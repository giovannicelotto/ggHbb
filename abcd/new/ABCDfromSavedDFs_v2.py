# %%
import pandas as pd
import sys
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
from plotDfs import plotDfs
from helpersABCD.abcd_maker_v2 import ABCD
from functions import getDfProcesses, cut, getDfProcesses_v2, cut_advanced
import numpy as np
import argparse
from helpersABCD.plot_v2 import pullsVsDisco, pullsVsDisco_pearson
import matplotlib.pyplot as plt

# %%
parser = argparse.ArgumentParser(description="Script.")
try:
    parser.add_argument("-m", "--modelName", type=str, help="e.g. Dec19_500p9", default=None)
    parser.add_argument("-dd", "--doubleDisco", type=bool, help="Single Disco (False) or double Disco (True). If false use jet1btag as variable", default=False)
    args = parser.parse_args()
    if args.modelName is not None:
        modelName = args.modelName
    dd = args.doubleDisco
except:
    print("Interactive mode")
    modelName = "Apr01_1000p0"
    dd = True
outFolder = "/t3home/gcelotto/ggHbb/PNN/resultsDoubleDisco/%s"%modelName
df_folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/%s"%modelName
bins = np.load(outFolder+"/mass_bins.npy")
#midpoints = (bins[:-1] + bins[1:]) / 2
#bins = np.sort(np.concatenate([bins, midpoints]))
#bins=bins[1:]
#bins[-1]=300
#bins=bins[:-1]
#midpoints = (bins[:-1] + bins[1:]) / 2
#bins = np.sort(np.concatenate([bins, midpoints]))

# %%
dfs = []
dfProcessesMC, dfProcessesData = getDfProcesses_v2()
# %%
# Loading data
dfsMC = []
isMCList = [0,
            1, 
            2,3,
            4,
            5,6,7,8, 9,10,
            11,12,13,
            14,15,16,17,18,
            19,20,21, 22, 35,
            36
            ]
for idx, p in enumerate(dfProcessesMC.process):
    if idx not in isMCList:
        continue
    df = pd.read_parquet(df_folder+"/df_%s%s_%s.parquet"%("dd_" if dd else "", p, modelName))
    dfsMC.append(df)
#dcor_MC_values =  dcor_plot_Data(dfsMC, dfProcessesMC.process, isMCList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcorMC_%s_%s.png"%(modelName, detail), nEvents=-1)
# %%
dfsData = []
isDataList = [
        #0,
#        1,
        2,
#       3,
        #4,
        #5,
           # 6
            ]

lumis = []
for idx, p in enumerate(dfProcessesData.process):
    if idx not in isDataList:
        continue
    df = pd.read_parquet(df_folder+"/dataframes%s%s_%s.parquet"%("_dd_" if dd else "", p, modelName))
    dfsData.append(df)
    lumi = np.load(df_folder+"/lumi%s%s_%s.npy"%("_dd_", p, modelName))
    lumis.append(lumi)
lumi = np.sum(lumis)
for idx, df in enumerate(dfsMC):
    dfsMC[idx].weight =dfsMC[idx].weight*lumi

dfsMC = cut(dfsMC, 'muon_pt', 9, None)
dfsData = cut(dfsData, 'muon_pt', 9, None)


# %%
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

#ax[0].set_xlabel("Bin")
#ax[0].set_ylabel("Normalized Count")
#ax[1].set_xlabel("Bin")
#ax[1].set_ylabel("Residual")
#ax[1].legend(bbox_to_anchor=(1,1))
plt.tight_layout()
#fig.savefig("/t3home/gcelotto/temp.png")

# %%
from scipy.stats import ks_2samp
import itertools
import seaborn as sns





# %%
# Function to apply KS test between two datasets
#def ks_test_between_datasets(df1, df2, feature):
#    stat, p_value = ks_2samp(df1[feature], df2[feature])
#    return p_value

# List to store results for each combination
#ks_pvalues_pnn1 = []
#ks_pvalues_pnn2 = []
# %%
# Loop over all pairs of datasets
#for idx, (n1, n2) in enumerate(itertools.combinations(range(len(dfsData)), 2)):
#    print("Combination %d %d"%(n1, n2))
#    p_value_pnn1 = ks_test_between_datasets(dfsData[n1], dfsData[n2], 'PNN1')
#    p_value_pnn2 = ks_test_between_datasets(dfsData[n1], dfsData[n2], 'PNN2')
#    print(p_value_pnn1, p_value_pnn2)
#    
#    ks_pvalues_pnn1.append(p_value_pnn1)
#    ks_pvalues_pnn2.append(p_value_pnn2)
# %%
#dataset_names = [dfProcessesData.process[isDataList].values[i] for i in range(len(dfsData))]
#pvalues_matrix_pnn1 = pd.DataFrame(index=dataset_names, columns=dataset_names)
#pvalues_matrix_pnn2 = pd.DataFrame(index=dataset_names, columns=dataset_names)
# %%
#for i, (n1, n2) in enumerate(itertools.combinations(range(len(dfsData)), 2)):
#    print(dataset_names[n1])
#    print(dataset_names[n2])
#    print(ks_pvalues_pnn1[i])
#    pvalues_matrix_pnn1.loc[dataset_names[n1], dataset_names[n2]] = ks_pvalues_pnn1[i]
#    pvalues_matrix_pnn2.loc[dataset_names[n1], dataset_names[n2]] = ks_pvalues_pnn2[i]


# %%
# Plot heatmap for PNN1 p-values using fig, ax
#fig, ax = plt.subplots(figsize=(10, 8))
#sns.heatmap(pvalues_matrix_pnn1.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1, ax=ax)
#ax.set_title('KS Test p-values for PNN1')
#plt.show()
#
#fig, ax = plt.subplots(figsize=(10, 8))
#sns.heatmap(pvalues_matrix_pnn2.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1, ax=ax)
#ax.set_title('KS Test p-values for PNN2')
#plt.show()

    # %%
x1 = 'PNN1' if dd else 'jet1_btagDeepFlavB'
x2 = 'PNN2' if dd else 'PNN'
xx = 'dijet_mass'
# tight WP for btag 0.7100
t1 = 0.25
t2 = 0.3

# %%
from helpersABCD.dcorPlot_process_datataking import dcor_plot_Data, dpearson_plot_Data


#dfsData = cut_advanced(dfsData, 'jet1_nTightMuons', 'abs(jet1_nTightMuons) < 1.1')
#dfsMC = cut_advanced(dfsMC, 'jet1_nTightMuons', 'abs(jet1_nTightMuons) < 1.1')
#dfsData = cut_advanced(dfsData, 'dijet_eta', 'abs(dijet_eta) > 3.0')
#dfsMC = cut_advanced(dfsMC, 'dijet_eta', 'abs(dijet_eta) > 3.0')

#dfsData = cut(dfsData, 'muon_pt', None, None)
#dfsMC = cut(dfsMC, 'muon_pt', None, None)

# %%
detail = ''
for f in dfProcessesData.process[isDataList].values:
    detail = detail + "_"+f[4:]
# %%
dcor_data_values =  dcor_plot_Data(dfsData, dfProcessesData.process, isDataList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcor_%s_%s.png"%(modelName, detail), nEvents=90000)
#dcor_MC_values =  dcor_plot_Data(dfsMC, dfProcessesMC.process, isMCList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcorMC_%s_%s.png"%(modelName, detail), nEvents=-1)
# %%
#df = pd.concat(dfsData)
#dcor_data_values =  dcor_plot_Data(dfsData, dfProcessesData.process, isDataList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dcor/dcor_%s_%s.png"%(modelName, detail), nEvents=-1)
#fig = plotDfs(dfsData=dfsData, dfsMC=dfsMC, isMCList=isMCList, dfProcesses=dfProcessesMC, nbin=101, lumi=lumi, log=True, blindPar=(True, 105, 30))
#dpearson_plot_Data([pd.concat(dfsData)], ['Data'], isDataList, bins, outFile="/t3home/gcelotto/ggHbb/abcd/new/plots/dpearsonR/pearson%s.png"%detail)
# %%
pulls_QCD_SR, err_pulls_QCD_SR = ABCD(dfsData, dfsMC,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=lumi, suffix='%s%s_%s'%(modelName, "_dd" if dd else "", detail), blindPar=(True, 100.5, 30))

# %%
from scipy.stats import pearsonr
pearson_data_values = []
df = pd.concat(dfsData)
# Parameters
num_bootstrap = 800  # Number of bootstrap resamples (Number of toys)
fraction = 1 / 5       # Use 1/5 of total events per bin

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
    
    # Bootstrap resampling
    n_samples = int(len(x) * fraction)  # Use 1/5 of data in each resample
    boot_corrs = []

    for _ in range(num_bootstrap):
        idx = np.random.choice(len(x), n_samples, replace=True)
        boot_corrs.append(pearsonr(x[idx], y[idx])[0])

    # Compute standard error and confidence interval
    se = np.std(boot_corrs)  
    ci_lower, ci_upper = np.percentile(boot_corrs, [2.5, 97.5])  
    plt.hist(np.clip(boot_corrs, -0.03, 0.03), bins=np.linspace(-0.03, 0.03, 101))
    plt.yscale('log')
    plt.show()
    plt.close()
    
    bootstrap_errors.append(se)
    confidence_intervals.append((ci_lower, ci_upper))

    # Print results for each bin
    print(f"mjj in ({b_low:.1f}, {b_high:.1f}): r = {pearson_coef:.5f}, SE = {se:.5f}, 95% CI = ({ci_lower:.5f}, {ci_upper:.5f})")

# Convert results to numpy arrays
pearson_data_values = np.array(pearson_data_values)
bootstrap_errors = np.array(bootstrap_errors)
confidence_intervals = np.array(confidence_intervals)

# %%

fig, ax = plt.subplots(1, 1)
maskBlind = pulls_QCD_SR>0
condition1 = pearson_data_values[maskBlind] > 0
condition2 = pulls_QCD_SR[maskBlind]-1 > 0
yTrue = pulls_QCD_SR[maskBlind]-1
ax.bar(((bins[:-1]+bins[1:])/2)[maskBlind], np.where(condition1, yTrue, -yTrue), width=1, label='Sign of pearson')
ax.bar(((bins[:-1]+bins[1:])/2)[maskBlind], np.where(condition2, yTrue*0.5, -yTrue*0.5), width=1, label='Sign of Pull')
ax.legend()
# %%
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
# %%
#bootstrap_errors=bootstrap_errors[1:]
#bins=bins[1:]
#pearson_data_values=pearson_data_values[1:]
#pulls_QCD_SR = pulls_QCD_SR[1:]
#err_pulls_QCD_SR = err_pulls_QCD_SR[1:]
maskBlind = (maskBlind) #& (abs(pulls_QCD_SR-1)<0.04)
(m_fit, q_fit), (m_err, q_err), cov_matrix_fit = pullsVsDisco_pearson(pearson_data_values, pulls_QCD_SR, err_pulls_QCD_SR, xerr=bootstrap_errors,mask =maskBlind, lumi=0, outName="/t3home/gcelotto/ggHbb/abcd/new/plots/pulls_vs_dcor/pulls_vs_dPearson_%s_%s.png"%(modelName, detail))
corrections = m_fit*pearson_data_values + q_fit
corrections = 1/corrections
# Compute uncertainty
# corrections = 1/(mx+q)
# dc/dm = -1/(mx+q)^2 * x
# dc/dq = -1/(mx+q)^2
der_m = -pearson_data_values/(m_fit*pearson_data_values+q_fit)**2
der_q = -1/(m_fit*pearson_data_values+q_fit)**2
err_corrections = np.sqrt(der_m**2*m_err**2 + der_q**2*q_err**2 + 2* der_m*der_q*cov_matrix_fit[1,0])
detailC = detail+'_corrected'
# %%
pulls_QCD_SR_new, err_pulls_QCD_SR_new = ABCD(dfsData, dfsMC,  x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi=lumi, suffix='%s_%s_%s'%(modelName, "dd" if dd else "", detailC), blindPar=(True, 120.5, 20), corrections=corrections, err_corrections=err_corrections)


# %%
print("New Uncertainties vs Old\n", err_pulls_QCD_SR_new/err_pulls_QCD_SR)
# %%
print("m : ", m_fit, " +- ", m_err)
print("q : ", q_fit, " +- ", q_err)
# %%




if False:
    fixed_sizes = [200000, 500000]

    # Store results
    pearson_results = {size: [] for size in fixed_sizes}
    pearson_results["all"] = []  # To store full bin statistics

    # Iterate over bins
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        m = (df.dijet_mass > b_low) & (df.dijet_mass < b_high)
        x_full = np.array(df.PNN1[m], dtype=np.float64)
        y_full = np.array(df.PNN2[m], dtype=np.float64)

        # Compute Pearson for fixed sizes
        for size in fixed_sizes:
            if len(x_full) < size:
                continue  # Skip if not enough events
            
            indices = np.random.choice(len(x_full), size, replace=False)  # Sample data
            x = x_full[indices]
            y = y_full[indices]

            pearson_coef = pearsonr(x, y)[0]
            pearson_results[size].append(pearson_coef)

        # Compute Pearson for ALL events in bin
        if len(x_full) > 1:  # Pearson requires at least two points
            pearson_results["all"].append(pearsonr(x_full, y_full)[0])
        else:
            pearson_results["all"].append(np.nan)  # If no data, avoid error

    # Convert results to arrays
    for key in pearson_results:
        pearson_results[key] = np.array(pearson_results[key])

    # Plot results
    plt.figure(figsize=(10, 6))
    for size in fixed_sizes:
        plt.plot(bins[:-1], pearson_results[size], marker='o', label=f"{size} events")
    plt.plot(bins[:-1], pearson_results["all"], marker='s', linestyle="--", color='black', label="All events")

    plt.xlabel("Dijet Mass Bin (GeV)")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.title("Pearson Correlation Convergence")
    plt.legend()
    plt.grid()
    plt.show()
# %%

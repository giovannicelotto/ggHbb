import ROOT
import sys
import yaml

# Load config
with open("/t3home/gcelotto/ggHbb/abcd/new/configABCD.yaml", "r") as f:
    config = yaml.safe_load(f)

modelName = config["modelName"]
variations = config["variations"]
hist_names = ["H", "ZJets", "QCD", "data_obs"]

# Keep file handles open
files = []
file_names = []

# Load files and print contents
for var in variations:
    detailC = config["detail"] + 'C_' + var
    file_path = f"/t3home/gcelotto/ggHbb/abcd/combineTry/shapes/counts_{modelName}_{detailC}.root"
    print(f"\n📂 Opening: {file_path}")
    
    file = ROOT.TFile(file_path, "READ")
    if not file or file.IsZombie():
        print(f"❌ Could not open file: {file_path}")
        continue

    files.append(file)
    file_names.append(file_path)

    print("📋 Contents:")
    for key in file.GetListOfKeys():
        print("   🔹", key.GetName())

# Output file
output_file = ROOT.TFile("/t3home/gcelotto/ggHbb/abcd/combineTry/shapes/merged_btagSystematics.root", "RECREATE")

# Copy histograms
def copy_histograms(files, file_names, suffixes):
    for file, file_path, suffix in zip(files, file_names, suffixes):
        print(f"\n🔄 Copying from: {file_path}")
        for name in hist_names:
            hist = file.Get(name)
            if hist:
                hist_clone = hist.Clone(f"{name}_{suffix}")
                hist_clone.Write()
                print(f"✅ Copied histogram: {name} -> {name}_{suffix}")
            else:
                print(f"⚠️ WARNING: Histogram '{name}' not found in {file_path}")

# Run the histogram copying
copy_histograms(files, file_names, suffixes=variations)
#copy_histograms(files, file_names, )
#copy_histograms(files, file_names, )

# Close files
for f in files:
    f.Close()
output_file.Close()

sys.exit()




# %%
means_B, stds_B, covs = [], [], []

for i in range(len(bins)-1):
    df_mass_bin = dfData[(dfData.dijet_mass > bins[i]) & (dfData.dijet_mass < bins[i+1])].reset_index(drop=True)
    print("Bin %d"%i)

    PNN1 = df_mass_bin.PNN1.to_numpy()
    PNN2 = df_mass_bin.PNN2.to_numpy()
    X1 = df_mass_bin[x1].to_numpy()
    X2 = df_mass_bin[x2].to_numpy()

    N_Bhats, corrections_list = [], []
    for _ in range(20000): 
        idx = np.random.choice(len(df_mass_bin), size=10000, replace=True)

        x1_sample = X1[idx]
        x2_sample = X2[idx]
        pnn1_sample = PNN1[idx]
        pnn2_sample = PNN2[idx]

        mA = (x1_sample < t1) & (x2_sample > t2)
        mC = (x1_sample < t1) & (x2_sample < t2)
        mD = (x1_sample > t1) & (x2_sample < t2)

        nA = mA.sum()
        nC = mC.sum()
        nD = mD.sum()

        if nC > 0:
            N_Bhats.append(nA * nD / nC)
            pearson = pearsonr(pnn1_sample, pnn2_sample)[0]
            corrections_list.append(1/(m_fit*pearson+q_fit))
    
    N_Bhats = np.array(N_Bhats)
    corrections_list = np.array(corrections_list)

    means_B.append(np.mean(N_Bhats))
    stds_B.append(np.std(N_Bhats))
    covs.append(np.cov(N_Bhats, corrections_list)[0, 1])  # Cov(B, rho)


# %%

                


coeffs = np.polyfit(means_B[1:], covs[1:], deg=2)
poly = np.poly1d(coeffs)

# Range per il fit
x_fit = np.linspace(min(means_B), max(means_B), 500)
y_fit = poly(x_fit)

# Plot con fig, ax
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(means_B[1:], covs[1:], marker='o', linestyle='none', label="Data")
ax.plot(x_fit, y_fit, color='orange', label=f"Fit: {coeffs[0]:.2e}·B² + {coeffs[1]:.2e}·B + {coeffs[2]:.2e}")

ax.set_xlabel("Mean B")
ax.set_ylabel("Cov(B, γ)")
#ax.set_title("Parabolic Fit of Cov(B, γ) vs Mean B")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()










# %%
# Try bootstrapping the error on B given A C D
mA      = (dfData[x1]<t1 ) & (dfData[x2]>t2 ) 
mB      = (dfData[x1]>t1 ) & (dfData[x2]>t2 ) 
mC      = (dfData[x1]<t1 ) & (dfData[x2]<t2 ) 
mD      = (dfData[x1]>t1 ) & (dfData[x2]<t2 ) 

mBin = (dfData[xx]>bins[4]) & (dfData[xx]<bins[5])
# %%
lambda_A = ((mBin) & (mA)).sum()
lambda_B = ((mBin) & (mB)).sum()
lambda_C = ((mBin) & (mC)).sum()
lambda_D = ((mBin) & (mD)).sum()



toys = {
    'A': np.random.poisson(lam=lambda_A, size=10000),
    'B': np.random.poisson(lam=lambda_B, size=10000),
    'C': np.random.poisson(lam=lambda_C, size=10000),
    'D': np.random.poisson(lam=lambda_D, size=10000)
}




# %%
plt.hist(toys["A"]*toys["D"]/toys["C"], bins=100)
plt.hist(toys["A"]*toys["D"]/toys["C"] / (q_fit + m_fit*pearson_data_values[4]), bins=100, histtype='step')
print(np.std(toys["A"]*toys["D"]/toys["C"]/ (q_fit + m_fit*pearson_data_values[4])))
print(lambda_A*lambda_D/lambda_C * np.sqrt(1/lambda_A  + 1/lambda_C  + 1/lambda_D))
plt.hist(toys["B"], histtype='step')

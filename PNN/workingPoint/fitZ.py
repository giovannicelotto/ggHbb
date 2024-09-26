# %%
import glob, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from functions import getXSectionBR, getZXsections
from functions import loadMultiParquet
import json

sys.path.append('/t3home/gcelotto/ggHbb/PNN')
from helpers.preprocessMultiClass import preprocessMultiClass
from helpers.getFeatures import getFeatures

import ROOT
from ROOT import RooFit, RooRealVar, RooDataSet, RooArgSet, RooArgList, RooGenericPdf, RooCBShape, RooAddPdf

nReal, nMC = 1, -1
xmin = 40
xmax = 300
bins = np.linspace(xmin, xmax, 100)





# %%
# Load Data

predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions"
isMCList = [0, 1, 36, 20, 21, 22, 23]
dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
processes = dfProcesses.process[isMCList].values

# Get predictions names path for both datasets
predictionsFileNames = []
for p in processes:
    predictionsFileNames.append(glob.glob(predictionsPath+"/%s/*.parquet"%p))


# %%
featuresForTraining, columnsToRead = getFeatures()
# extract fileNumbers
predictionsFileNumbers = []

for isMC, p in zip(isMCList, processes):
    idx = isMCList.index(isMC)
    print("Process %s # %d"%(p, isMC))
    l = []
    for fileName in predictionsFileNames[idx]:
        fn = re.search(r'fn(\d+)\.parquet', fileName).group(1)
        l.append(int(fn))

    predictionsFileNumbers.append(l)

# %%  


#paths = list(dfProcesses.flatPath.values[isMCList])
paths =["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
        ]
dfs= []
print(predictionsFileNumbers)
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'jet1_qgl', 'jet2_qgl', 'dijet_dR', 'dijet_dPhi', 'jet3_mass', 'jet3_qgl', 'Pileup_nTrueInt'], returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers, returnFileNumberList=True)


# %%
preds = []
predictionsFileNamesNew = []
for isMC, p in zip(isMCList, processes):
    idx = isMCList.index(isMC)
    print("Process %s # %d"%(p, isMC))
    l =[]
    for fileName in predictionsFileNames[idx]:
        print(fileName)
        fn = int(re.search(r'fn(\d+)\.parquet', fileName).group(1))
        if fn in fileNumberList[idx]:
            l.append(fileName)
    predictionsFileNamesNew.append(l)
    
    print(len(predictionsFileNamesNew[idx]), " files for process")
    df = pd.read_parquet(predictionsFileNamesNew[idx])
    preds.append(df)


# given the fn load the data


# preprocess 
dfs = preprocessMultiClass(dfs=dfs)
# %%
for idx, df in enumerate(dfs):
    print(idx)
    dfs[idx]['PNN'] = np.array(preds[idx])





# %%
# Working Points
# Define the working Points (chosen to maximize Significance) and filter the dataframes
W_Z = []
for idx, df in enumerate(dfs[2:]):
    w = df.sf*getZXsections()[idx]/numEventsList[idx+2]*nReal*0.774/1017*1000
    W_Z.append(w)


# %%
dfZ = pd.concat(dfs[2:])
W_Z=np.concatenate(W_Z)

# %%
workingPoint = (dfs[0].PNN > 0.696) & (dfs[0].dijet_mass>xmin)
workingPoint_Z = (dfZ.PNN>0.696)& (dfZ.dijet_mass>xmin)
mass = dfs[0].dijet_mass[workingPoint]
mass_Z = dfZ.dijet_mass[workingPoint_Z]
weights_Z = W_Z[workingPoint_Z]



































# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
x = (bins[1:] + bins[:-1])/2
counts = np.histogram(mass,  bins=bins)[0]
errors = np.sqrt(counts)
integral = np.sum(counts * np.diff(bins))

counts_Z = np.histogram(mass_Z, bins=bins, weights=weights_Z)[0]
errors_Z = np.sqrt(np.histogram(mass_Z, bins=bins, weights=weights_Z**2)[0])

ax[0].errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, color='black', linestyle='none')
ax[0].hist(bins[:-1], bins=bins, weights=counts_Z*10, histtype=u'step', color='red', label='Z MC')[0]
ax[0].set_xlim(xmin, xmax)
ax[0].set_xlabel("Dijet Mass [GeV]")
# Z MC
ax[1].errorbar(x, counts_Z, yerr=errors_Z, color='red', linestyle='none')
ax[1].set_xlim(xmin, xmax)
ax[1].set_xlabel("Dijet Mass [GeV]")




# %%
x = RooRealVar("x", "mass", 40, 200)
w = RooRealVar("w", "weight", 0.0, 5.0)

dataset = RooDataSet("dataset", "dataset with weights", RooArgSet(x, w), RooFit.WeightVar(w))

# Fill the dataset with values and corresponding weights
for val, weight in zip(mass_Z, weights_Z):
    x.setVal(val)
    w.setVal(weight)
    dataset.add(RooArgSet(x, w), weight)

# %%
mean = RooRealVar("mean", "mean", 93, 86, 98)
sigma_left = RooRealVar("sigma_left", "sigma (left)", 13.3, 9, 20)
alpha_left = RooRealVar("alpha_left", "alpha (left)", 1.17, 1, 10)
n_left = RooRealVar("n_left", "n (left)", 3.1, 0.1, 30)

# Parameters for the right Crystal Ball
sigma_right = RooRealVar("sigma_right", "sigma (right)", 15, 9, 20)
alpha_right = RooRealVar("alpha_right", "alpha (right)", -1.576, -10, -0.1)
n_right = RooRealVar("n_right", "n (right)", 30, 0.1, 50)

# Create the two Crystal Ball components
crystal_left = RooCBShape("crystal_left", "Crystal Ball (left)", x, mean, sigma_left, alpha_left, n_left)
crystal_right = RooCBShape("crystal_right", "Crystal Ball (right)", x, mean, sigma_right, alpha_right, n_right)

# Combine the two with a RooAddPdf, assuming equal fractions (0.5 for each side)
frac = RooRealVar("frac", "fraction of left CB", 0.5, 0.0, 1.0)
double_cb = RooAddPdf("double_cb", "Double-sided Crystal Ball", RooArgList(crystal_left, crystal_right), RooArgList(frac))

# Fit the model to the dataset
fit_result = double_cb.fitTo(dataset, RooFit.SumW2Error(True), RooFit.Save())
corr_matrix = fit_result.correlationMatrix()
corr_matrix.Print()

# Print the covariance matrix as well
cov_matrix = fit_result.covarianceMatrix()
cov_matrix.Print()
# %%
# Plotting cell

frame = x.frame(RooFit.Title("Double-Sided Crystal Ball Fit"))

# Plot the dataset on this frame
dataset.plotOn(frame, RooFit.MarkerColor(ROOT.kBlack))

# Plot the double-sided Crystal Ball fit result on the same frame
double_cb.plotOn(frame, RooFit.LineColor(ROOT.kRed))

# Create a canvas to draw the plot
c = ROOT.TCanvas("c", "c", 800, 600)
frame.Draw()

# Calculate chi2 and ndof
chi2 = frame.chiSquare()  # This gives chi2/ndof

# Add chi2 and ndof information on the plot
chi2_text = ROOT.TLatex()
chi2_text.SetNDC()  # Use normalized device coordinates (NDC)
chi2_text.SetTextSize(0.04)  # Text size in the plot
chi2_text.SetTextAlign(33)  # Align at top-left corner

# Text to display
chi2_label = f"#chi^{{2}}/NDOF = {chi2:.3f}"

# Specify the position of the text box (x, y)
chi2_text.DrawLatex(0.85, 0.85, chi2_label)
c.Draw()
# %%
print(f"Mean: {mean.getVal()} ± {mean.getError()}")
print(f"Sigma Left: {sigma_left.getVal()} ± {sigma_left.getError()}")
print(f"Alpha Left: {alpha_left.getVal()} ± {alpha_left.getError()}")
print(f"N Left: {n_left.getVal()} ± {n_left.getError()}")

print(f"Sigma Right: {sigma_right.getVal()} ± {sigma_right.getError()}")
print(f"Alpha Right: {alpha_right.getVal()} ± {alpha_right.getError()}")
print(f"N Right: {n_right.getVal()} ± {n_right.getError()}")

print(f"Fraction of Left CB: {frac.getVal()} ± {frac.getError()}")
# %%

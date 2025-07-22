# %%
import ROOT
ROOT.gSystem.CompileMacro("/t3home/gcelotto/ggHbb/newFit/rooFit/helpersFunctions/RooDoubleCB.cc", "kf")
print("Compiled")
# Open the file
# Load the workspace
f = ROOT.TFile.Open("/t3home/gcelotto/ggHbb/newFit/rooFit/workspace_sig.root")
w = f.Get("workspace_sig")
# %%
# %%
# Prepare variables, PDFs, and datasets for both categories
cats = [1, 2]
datahists = {}
pdfs = {}
mass_vars = {}

# Set ranges and extract everything needed
for cat in cats:
    pdfs[cat] = w.pdf(f"model_cat{cat}")
    datahists[cat] = w.data(f"rooHist_data_cat{cat}")
    mass_vars[cat] = w.var(f"dijet_mass_c{cat}")

    # Set ranges for integrals/normalization
    if cat == 1:
        t0, t1, t2, t3 = 50, 110, 140, 300
    elif cat == 2:
        t0, t1, t2, t3 = 50, 129.9, 130, 250
    mass_vars[cat].setRange("R1", t0, t1)
    mass_vars[cat].setRange("R2", t2, t3)

# %%
# Create a RooCategory to label the two channels




# %%
# Construct a simultaneous PDF

# %%
channelCat = ROOT.RooCategory("channelCat", "Category")
channelCat.defineType("cat1")
channelCat.defineType("cat2")
Dtpwfd6dC
simPdf = ROOT.RooSimultaneous("simPdf", "simultaneous model", channelCat)
simPdf.addPdf(pdfs[1], "cat1")
simPdf.addPdf(pdfs[2], "cat2")
# Define observable sets for each category
obs_c1 = ROOT.RooArgSet(mass_vars[1])  # dijet_mass_c1
obs_c2 = ROOT.RooArgSet(mass_vars[2])  # dijet_mass_c2

# Construct the simultaneous dataset with explicit RooFit.Index and Import
combinedData = ROOT.RooDataHist(
    "combinedData", "combinedData",
    ROOT.RooArgSet(mass_vars[1], mass_vars[2]),  # This works when vars are different across cats
    ROOT.RooFit.Index(channelCat),
    ROOT.RooFit.Import("cat1", datahists[1]),
    ROOT.RooFit.Import("cat2", datahists[2])
)



# %%
# Fit the simultaneous model to the combined dataset
fit_result = simPdf.fitTo(combinedData, ROOT.RooFit.Save(), ROOT.RooFit.Extended(True), ROOT.RooFit.PrintLevel(1))

# %%
# Print or save fit results
fit_result.Print()

# Need to run with CMSENV
# /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src

import ROOT

# Open the input file
input_file = ROOT.TFile.Open("/t3home/gcelotto/ggHbb/newFit/rooFit/workspace_sig.root")
if not input_file or input_file.IsZombie():
    raise RuntimeError("Failed to open the input ROOT file.")

# Retrieve the workspace
workspace = input_file.Get("workspace_sig")
if not workspace:
    raise RuntimeError("Workspace 'workspace_sig' not found in the file.")

# Retrieve the PDFs from the workspace
model_f0_cat2 = workspace.obj("model_f0_cat2")
model_f1_cat2 = workspace.obj("model_f1_cat2")
#model_f16_cat2 = workspace.obj("model_f16_cat2")
model_H_c2    = workspace.obj("model_H_c2")
data_obs = workspace.obj("rooHist_data_cat2")  # Replace "data_obs" with the actual name of your RooDataHist

nQCD_cat2= workspace.obj("nQCD_cat2")
nZ_times_mu_c2= workspace.obj("nZ_times_mu_c2")
val_QCD = nQCD_cat2.getVal()
val_Z   = nZ_times_mu_c2.getVal()
sum_val = val_QCD + val_Z
multipdf_norm = ROOT.RooRealVar("multipdf_norm", "Sum of QCD and Z", sum_val)

r = ROOT.RooRealVar("r", "signal strength Higgs", 1, -10, 10)

if not all([model_f0_cat2, model_f1_cat2, model_H_c2, data_obs]):
    raise RuntimeError("One or more model PDFs not found in the workspace.")


# Create a RooCategory to index the PDFs
pdf_index = ROOT.RooCategory("pdfindex", "PDF index")

# Create the RooArgList of PDFs
pdf_list = ROOT.RooArgList()
pdf_list.add(model_f0_cat2)
pdf_list.add(model_f1_cat2)
#pdf_list.add(model_f16_cat2)

# Create the RooMultiPdf
multipdf = ROOT.RooMultiPdf("multipdf", "Background MultiPdf", pdf_index, pdf_list)

# Create a new workspace
new_workspace = ROOT.RooWorkspace("workspace_step2", "workspace_step2")

# Import RooMultiPdf and pdf_index
getattr(new_workspace, "import")(multipdf, ROOT.RooFit.RecycleConflictNodes())
#getattr(new_workspace, "import")(pdf_index)

# Import signal model as "signal"
getattr(new_workspace, "import")(model_H_c2)
getattr(new_workspace, "import")(data_obs)
getattr(new_workspace, "import")(multipdf_norm)
getattr(new_workspace, "import")(r)

# Import the observable (assuming dijet_mass_c2)
#dijet_mass_c2 = workspace.obj("dijet_mass_c2")x    
#getattr(new_workspace, "import")(dijet_mass_c2)

# Save the new workspace to a new ROOT file
output_file = ROOT.TFile("/t3home/gcelotto/ggHbb/newFit/rooFit/workspace_step2_Aug28.root", "RECREATE")
new_workspace.Write()
output_file.Close()

print("MultiPdf saved in workspace_step2.root successfully.")

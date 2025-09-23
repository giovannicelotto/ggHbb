# Need to run with CMSENV
# /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src

import ROOT
import argparse
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=int, default="2", help='Config File')
if hasattr(sys, 'ps1') or not sys.argv[1:]:
    # Interactive mode (REPL, Jupyter) OR no args provided → use defaults
    args = parser.parse_args([])
else:
    # Normal CLI usage
    args = parser.parse_args()


# Open the input file
input_file = ROOT.TFile.Open("/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws%d.root"%(int(args.config)))
if not input_file or input_file.IsZombie():
    raise RuntimeError("Failed to open the input ROOT file.")

# Retrieve the workspace
workspace = input_file.Get("workspace_sig")
if not workspace:
    raise RuntimeError("Workspace 'workspace_sig' not found in the file.")





print("\n"*10)
workspace.Print()
print("\n"*10)
model_Z_c    = workspace.obj("model_Z_c%d"%(int(args.config)))
data_obs = workspace.obj("rooHist_data_cat%d"%(args.config))  # Replace "data_obs" with the actual name of your RooDataHist
nZ_times_mu = workspace.obj("nZ_times_mu")  
print(type(data_obs))
data_obs.Print()



#r = ROOT.RooRealVar("r", "signal strength Higgs", 1, -10, 10)


# Create a new workspace
new_workspace = ROOT.RooWorkspace("workspace_step2", "workspace_step2")
getattr(new_workspace, "import")(model_Z_c)
getattr(new_workspace, "import")(data_obs)
getattr(new_workspace, "import")(nZ_times_mu)
#getattr(new_workspace, "import")(r)


# Save the new workspace to a new ROOT file
output_file = ROOT.TFile("/t3home/gcelotto/ggHbb/WSFit/ws/step2/ws%d.root"%(int(args.config)), "RECREATE")
new_workspace.Write()
output_file.Close()

print("MultiPdf saved in workspace_step2.root successfully.")

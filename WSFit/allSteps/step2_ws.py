# Prepare the workspace to be used in the FTest

# Need to run with CMSENV
# /t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kWarning
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
input_file = ROOT.TFile.Open("/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws%d_nominal.root"%(int(args.config)))
if not input_file or input_file.IsZombie():
    raise RuntimeError("Failed to open the input ROOT file.")

# Retrieve the workspace
workspace = input_file.Get("WS")
if not workspace:
    raise RuntimeError("Workspace 'ws' not found in the file.")
ROOT.SetOwnership(workspace, False)



model_Z_c    = workspace.obj("model_Z_c%d"%(int(args.config)))
ROOT.SetOwnership(model_Z_c, False)
data_obs = workspace.obj("rooHist_data_cat%d"%(args.config))  # Replace "data_obs" with the actual name of your RooDataHist
ROOT.SetOwnership(data_obs, False)
#nZ_times_mu = workspace.obj("nZ_times_mu")  
#ROOT.SetOwnership(nZ_times_mu, False)
#print(type(data_obs))
#data_obs.Print()
# Create a new workspace
new_workspace = ROOT.RooWorkspace("workspace_step2", "workspace_step2")
for idx, obj in enumerate([model_Z_c, data_obs, ]):
    if not obj:
        raise RuntimeError("Missing object ", idx)
    ROOT.SetOwnership(obj, False)
getattr(new_workspace, "import")(model_Z_c)
#print("[DEBUG] Reached here")
getattr(new_workspace, "import")(data_obs)
#getattr(new_workspace, "import")(nZ_times_mu)
#getattr(new_workspace, "import")(r)


# Save the new workspace to a new ROOT file
output_file = ROOT.TFile("/t3home/gcelotto/ggHbb/WSFit/ws/step2/ws%d.root"%(int(args.config)), "RECREATE")
new_workspace.Write()
output_file.Close()

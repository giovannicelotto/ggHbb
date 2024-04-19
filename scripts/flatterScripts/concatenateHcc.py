import ROOT

# Set the path to the directory containing your root files
directory_path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggHcc2023Dec14/GluGluHToCC_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8/crab_GluGluHToCC/231214_162727/0000"

# Create a TChain and add the 'Events' tree from each file
chain = ROOT.TChain("Events")
files = ROOT.TFileCollection()
files.Add(directory_path + "/*.root")
chain.AddFileInfoList(files.GetList())

# Set the path for the output concatenated file
output_path = directory_path+"/HccAlone.root"

# Create a new output file
output_file = ROOT.TFile(output_path, "recreate")

# Clone the tree and copy the entries
output_tree = chain.CloneTree(-1, "fast")

# Write and close the output file
output_file.Write()
output_file.Close()

print("Files concatenated and saved to:", output_path)

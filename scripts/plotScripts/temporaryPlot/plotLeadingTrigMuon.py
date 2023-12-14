import uproot
import ROOT
import glob
import numpy as np

path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/000*"
fileNames = glob.glob(path + "/*.root")

canvas = ROOT.TCanvas("canvas", "Histogram Canvas", 800, 600)
histogram = ROOT.TH1F("histogram", "", 50, 0, 25)
histogram.GetXaxis().SetTitle("Leading trigger muon p_{T}")
histogram.GetYaxis().SetTitle("Events")
N_nano = np.load("/t3home/gcelotto/ggHbb/outputs/N_BPH_Nano.npy")
totalEntries = 0

for fileName in fileNames[:]:
    print("%d/%d"%(fileNames.index(fileName)+1, len(fileNames)), fileName, "\n")
    f = uproot.open(fileName)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries 
    totalEntries = totalEntries + maxEntries
    for ev in  range(maxEntries):
    
        Muon_pt                     = branches["Muon_pt"][ev]
        Muon_isTriggering           = branches["Muon_isTriggering"][ev]
        Muon_fired_HLT_Mu9_IP6      = branches["Muon_fired_HLT_Mu9_IP6"][ev]
        nMuon                       = branches["nMuon"][ev]
        for mu in range(nMuon):
            #if Muon_isTriggering[mu]==1:
            if Muon_fired_HLT_Mu9_IP6[mu]==1:
                histogram.Fill(Muon_pt[mu])
                break
    

    histogram.Scale(1./histogram.Integral(0, histogram.GetNbinsX()+1)*N_nano*histogram.GetEntries()/totalEntries)
    histogram.Draw("hist")
    canvas.SetLogy()
    canvas.SaveAs("/t3home/gcelotto/ggHbb/outputs/plots/HLTMu9IP6_pt.pdf")

    root_file = ROOT.TFile("/t3home/gcelotto/ggHbb/outputs/HLTMu9IP6.root", "RECREATE")
    histogram.Write()
    root_file.Close()
    print("Entries : %d +- %d"%(histogram.GetEntries(), np.sqrt(histogram.GetEntries())))
    print("Total   : %d +- %d"%(totalEntries, np.sqrt(totalEntries)))
    print("Ratio   : %.3f"%(histogram.GetEntries()/totalEntries))



## Draw the histogram
#histogram.Draw()
#
## Show the canvas
#canvas.Draw()
#
## Keep the program running to display the canvas
#ROOT.gApplication.Run()

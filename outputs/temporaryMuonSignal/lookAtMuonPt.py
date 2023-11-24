#-6
import numpy as np
import glob
import ROOT
ROOT.gStyle.SetOptFit(111)
path = "/t3home/gcelotto/ggHbb/outputs/temporaryMuonSignal"
npyFileNames = glob.glob(path+"/*.npy")
h = ROOT.TH1F("h", ";Muon p_{T} [GeV]; Events", 50, 0, 50)
for npyFileName in npyFileNames:
    f = np.load(npyFileName)
    for ev in range(f.shape[0]):
        h.Fill(f[ev,-6])

canvas = ROOT.TCanvas("my_canvas", "My Canvas", 800, 600)

h.Draw()
h.GetYaxis().SetRangeUser(0, 2500)

# Fit
exponential_function = ROOT.TF1("exponential_function", "[0]*exp(-x/[1])", 7, 50)
exponential_function.SetParameters(2900, 10)  # Set initial parameters (amplitude, decay constant)
#exponential_function.Draw("same")
# Fit the histogram with the exponential function
h.Fit("exponential_function", "R")

canvas.SaveAs("/t3home/gcelotto/ggHbb/outputs/temporaryMuonSignal/output.png")


# %%
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mplhep as hep
from scipy.stats import chi2
hep.style.use("CMS")
import sys
from extractXYErr_fromRooDataHist import *
ROOT.gSystem.CompileMacro("/t3home/gcelotto/ggHbb/newFit/rooFit/helpersFunctions/RooDoubleCB.cc", "kf")

# Argument parsing
parser = argparse.ArgumentParser(
    prog='fit',
    description='Fit more categories simultaneously',
    epilog='Text at the bottom of help'
)
parser.add_argument('-c', '--cat', type=int, default=2)
parser.add_argument('-f', '--function', type=int, nargs='+', default=[0, 1])
parser.add_argument('-fit', '--fit', type=int, default=1)
parser.add_argument('-chi2', '--chi2', type=int, default=0)

if __name__ == '__main__' and not hasattr(sys, 'ps1'):
    args = parser.parse_args()
else:
    # Interactive mode
    args = parser.parse_args([])

# %%
# Open ROOT file and extract workspace
f = ROOT.TFile.Open("/t3home/gcelotto/ggHbb/newFit/rooFit/workspace_sig.root")
w = f.Get("workspace_sig")

x = w.var(f"dijet_mass_c{args.cat}")
data_hist = w.data(f"rooHist_data_cat{args.cat}")

# Extract and fit PDFs
colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2, ROOT.kMagenta]
pdfs = []

for i, func_id in enumerate(args.function):
    pdf = w.pdf(f"f{func_id}_c{args.cat}")
    if not pdf:
        raise RuntimeError(f"PDF f{func_id}_c{args.cat} not found")
    
    # Fit the PDF to the data
    fit_result = pdf.fitTo(data_hist, ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))
    pdfs.append((func_id, pdf, fit_result))
    print(f"Fit completed for f{func_id}_c{args.cat}")

# %%
# Plot using RooPlot
c1 = ROOT.TCanvas("c1", "", 800, 600)

frame = x.frame()
frame.SetTitle("")
data_hist.plotOn(frame, ROOT.RooFit.Name("data"))
# Retrieve the plotted object by name
hist_obj = frame.findObject("data")

# Change marker size
if hist_obj:
    hist_obj.SetMarkerSize(0.5)  # Default is 1.0, increase for larger markers
for i, (func_id, pdf, _) in enumerate(pdfs):
    pdf.chi2FitTo(data_hist,
                      ROOT.RooFit.SumW2Error(True), 
                       ROOT.RooFit.Save(),
                       Range="R1,R2",)
    pdf.plotOn(frame, ROOT.RooFit.LineColor(colors[i % len(colors)]), ROOT.RooFit.Name(f"f{func_id}"))
legend = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.SetTextSize(0.03)

# Add data to legend
legend.AddEntry(frame.findObject("data"), "Data", "lep")

# Add each PDF curve to the legend
for i, (func_id, pdf, _) in enumerate(pdfs):
    obj = frame.findObject(f"f{func_id}")
    if obj:
        legend.AddEntry(obj, f"f{func_id}", "l")  # "l" = line
frame.Draw()
legend.Draw()
c1.SaveAs("/t3home/gcelotto/ggHbb/newFit/rooFit/doubleFit.png")
# %%

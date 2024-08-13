
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import loadMultiParquet, cut
import mplhep as hep
hep.style.use("CMS")
from array import array
import ROOT
def combined_fit(x, params):
    # Crystal Ball parameters
    alpha = params[0]
    n = params[1]
    mean = params[2]
    sigma = params[3]
    normCB = params[4]
    
    # Bernstein polynomial parameters
    p0 = params[5]
    p1 = params[6]
    p2 = params[7]
    
    # One-sided Crystal Ball function
    crystal_ball = ROOT.Math.crystalball_function(x[0], alpha, n, mean, sigma)*normCB
    
    # Second-order Bernstein polynomial
    #bernstein_poly = p0 * (1 - x[0])**2 + 2 * p1 * x[0] * (1 - x[0]) + p2 * x[0]**2
    pol = p0  + p1 * x[0] + p2*x[0]*x[0]
    
    return crystal_ball + pol

paths = [
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/**/"
        ]

xsections = [1]
labels = ['GluGluHToBB']

dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=-2, nMC=-1, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt'], returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)

dfs = cut(dfs, 'jet1_pt', 20, None)
dfs = cut(dfs, 'jet2_pt', 20, None)

#fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 300, 100)
#for idx, df in enumerate(dfs):
#    counts = np.histogram(df.dijet_mass, bins=bins, weights=df.sf*xsections[idx]/numEventsList[idx])[0]
#    ax.hist(bins[:-1], bins=bins, weights=counts, label = labels[idx])
#hep.cms.label()
#ax.legend()
#outName = "/t3home/gcelotto/ggHbb/outputs/plots/fits/zpeak.png"
#fig.savefig(outName)
#print("saved in", outName)

ROOT.gStyle.SetOptStat(0000)
ROOT.gStyle.SetOptFit(1111)


c1 = ROOT.TCanvas("c1", "c1", 800, 600)
hist = ROOT.TH1F("hist", "", len(bins) - 1, bins)

hists = [
hist
]
colors = [ROOT.kBlue]
hs = ROOT.THStack("hs", "Stacked Histograms")
print("For started")
for idx, df in enumerate(dfs):
    counts = np.histogram(df.dijet_mass, bins=bins, weights=df.sf)[0]
    for i in range(len(counts)):
        hists[idx].SetBinContent(i + 1, counts[i])
    hists[idx].SetFillColor(colors[idx])
    print(hists[0].GetBinContent(10))
    print(hists[0].GetBinError(10))
    #hists[idx].Scale(xsections[idx]/numEventsList[idx])
    hs.Add(hists[idx])

combined_hist = hists[0].Clone("combined_hist")

fit_bkg = ROOT.TF1("fit_bkg", "[0] + [1]*x + [2]*x*x", 20, 300)
fit_bkg.SetParameters(0.1,0.01, 0.001)
combined_hist.Fit(fit_bkg, "RE","",180, 300)

# Create the TF1 object
fit_function = ROOT.TF1("fit_function", combined_fit, 0, 300, 8)

# Set initial parameter values for the fit (example values, may need adjustment)
fit_function.SetParameters(1., 2, 17, 125, 90e3, fit_bkg.GetParameter(0), fit_bkg.GetParameter(1), fit_bkg.GetParameter(2))
fit_function.SetParLimits(0, 0, 10)
fit_function.SetParLimits(1, 0, 30)
fit_function.SetParLimits(2, 10, 20)
fit_function.SetParLimits(3, 115, 130)
fit_function.SetParLimits(4, 0, 20e4)


# Fit the combined histogram
print("Fit started")
fit_function.SetParNames(r"#alpha", "n",r"#sigma",r"#mu","Norm", "p0","p1","p2")
combined_hist.Fit(fit_function, "RE+","",10, 300)
pol2 = ROOT.TF1("pol2", "[0] + [1]*x + [2]*x*x", 10, 300)
pol2.SetParameters(fit_function.GetParameter(5), fit_function.GetParameter(6), fit_function.GetParameter(7))
pol2.SetLineColor(ROOT.kGreen)


legend = ROOT.TLegend(0.15, 0.7, 0.35, 0.8)
for idx, label in enumerate(labels):
    legend.AddEntry(hists[idx], label, "f")

combined_hist.GetXaxis().SetTitle("Dijet Mass [GeV]")
combined_hist.GetYaxis().SetTitle("Counts [a.u.]")
combined_hist.Draw("hist")

hs.Draw("hist same")
pol2.Draw("same")
fit_function.SetLineColor(ROOT.kRed)
fit_function.Draw("same")
legend.Draw()


#c1.Update()
c1.Draw()
c1.SaveAs("/t3home/gcelotto/ggHbb/outputs/plots/fits/higgs.png")
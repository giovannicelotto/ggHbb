
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import loadMultiParquet, cut
import mplhep as hep
hep.style.use("CMS")
from array import array
import ROOT
def DoubleSidedCrystalballFunction(x, par):
    alpha_l = par[0]
    alpha_r = par[1]
    n_l = par[2]
    n_r = par[3]
    mean = par[4]
    sigma = par[5]
    N = par[6]
    
    t = (x[0] - mean) / sigma
    result = 0.0
    
    fact1TLessMinosAlphaL = alpha_l / n_l
    fact2TLessMinosAlphaL = (n_l / alpha_l) - alpha_l - t
    
    fact1THihgerAlphaH = alpha_r / n_r
    fact2THigherAlphaH = (n_r / alpha_r) - alpha_r + t
    
    if -alpha_l <= t <= alpha_r:
        result = np.exp(-0.5 * t * t)
    elif t < -alpha_l:
        result = np.exp(-0.5 * alpha_l * alpha_l) * np.power(fact1TLessMinosAlphaL * fact2TLessMinosAlphaL, -n_l)
    elif t > alpha_r:
        result = np.exp(-0.5 * alpha_r * alpha_r) * np.power(fact1THihgerAlphaH * fact2THigherAlphaH, -n_r)
    
    return N * result

paths = [
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/**/"
        ]

xsections = [1]
labels = ['GluGluHToBB']

dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=-2, nMC=-1, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt'], returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)

dfs = cut(dfs, 'jet1_pt', 20, None)
dfs = cut(dfs, 'jet2_pt', 20, None)

bins = np.linspace(0, 300, 80)
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
    #hists[idx].Scale(xsections[idx]/numEventsList[idx])
    hs.Add(hists[idx])

combined_hist = hists[0].Clone("combined_hist")
# Create the TF1 object
fitFunction = ROOT.TF1("fitFunction", DoubleSidedCrystalballFunction, 0, 300, 7)
fitFunction.SetNpx(5000)
fitFunction.SetParameters(0.5, 2, 1, 1, 125, 15, 45000)
fitFunction.SetParLimits(0, 0, 5)
fitFunction.SetParLimits(1, 0, 5)
#fitFunction.SetParLimits(2, 0, 10)
fitFunction.SetParLimits(3, 0, 40)
fitFunction.SetParLimits(4, 110, 145)
fitFunction.SetParLimits(5, 10, 30)
fitFunction.SetParLimits(6, 0, 100e3)




# Fit the combined histogram
fitFunction.SetParNames("#alpha_l", "#alpha_r", "n_l", "n_r", "#mu", "#sigma", "N")
combined_hist.Fit(fitFunction, "RE+","",80, 250)

legend = ROOT.TLegend(0.15, 0.7, 0.35, 0.8)
for idx, label in enumerate(labels):
    legend.AddEntry(hists[idx], label, "f")

combined_hist.GetXaxis().SetTitle("Dijet Mass [GeV]")
combined_hist.GetYaxis().SetTitle("Counts [a.u.]")
combined_hist.Draw("hist")

hs.Draw("hist same")
#pol2.Draw("same")
fitFunction.SetLineColor(ROOT.kRed)
fitFunction.Draw("same")
legend.Draw()


gaus = ROOT.TF1("gaus", "gaus(0)", 10, 300, 3)
gaus.SetParameters(fitFunction.GetParameter(6), fitFunction.GetParameter(4), fitFunction.GetParameter(5))
gaus.SetLineColor(ROOT.kGray)
gaus.Draw("same")


#c1.Update()
c1.Draw()
c1.SaveAs("/t3home/gcelotto/ggHbb/outputs/plots/fits/higgs.png")
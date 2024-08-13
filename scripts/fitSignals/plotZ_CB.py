
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import loadMultiParquet, cut
import mplhep as hep
hep.style.use("CMS")
from array import array
import ROOT
from ROOT import RooFit, RooRealVar, RooDataSet, RooPlot, RooCrystalBall, RooArgList
import cmsstyle as CMS
import sys
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
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJets",
            ]

xsections = [5.261e+03, 1012, 114.2, 25.34, 12.99, 9.8, ]
labels = ['ZJets100-200', 'ZJets200-400', 'ZJets400-600', 'ZJets600-800', 'ZJets800-Inf','EWKZJets']

dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=-2, nMC=-1, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt'], returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)

dfs = cut(dfs, 'jet1_pt', 20, None)
dfs = cut(dfs, 'jet2_pt', 20, None)
#dfs = cut(dfs, 'dijet_pt', 200, None)


xmin, xmax = 20, 350
bins = np.linspace(xmin, xmax, 40)

ROOT.gStyle.SetOptStat(0000)
ROOT.gStyle.SetOptFit(1111)



c1 = ROOT.TCanvas("c1", "c1", 800, 600)
CMS.SetExtraText("Simulation")
CMS.SetLumi("")
CMS.SetEnergy("13")
#c1 = CMS.cmsCanvas('', xmin, xmax, 0, 9, 'Dijet Mass [GeV]', 'Counts [a.u.]', square = CMS.kSquare, extraSpace=0.01, iPos=0)
#c1.Draw()



hist100 = ROOT.TH1F("hist100", "", len(bins) - 1, bins)
hist200 = ROOT.TH1F("hist200", "", len(bins) - 1, bins)
hist400 = ROOT.TH1F("hist400", "", len(bins) - 1, bins)
hist600 = ROOT.TH1F("hist600", "", len(bins) - 1, bins)
hist800 = ROOT.TH1F("hist800", "", len(bins) - 1, bins)
histEWK = ROOT.TH1F("histEWK", "", len(bins) - 1, bins)

hists = [hist100 ,hist200 ,hist400 ,hist600 ,hist800,histEWK]
colors = [ROOT.kBlue-5, ROOT.kBlue-10, ROOT.kGreen, ROOT.kOrange+7, ROOT.kMagenta, ROOT.kYellow+2]
hs = ROOT.THStack("hs", "Stacked Histograms")
print("For started")
for idx, df in enumerate(dfs):
    counts = np.histogram(df.dijet_mass, bins=bins, weights=df.sf)[0]
    for i in range(len(counts)):
        hists[idx].SetBinContent(i + 1, counts[i])
    hists[idx].SetFillColor(colors[idx])
    print(labels[idx], hists[idx].GetBinContent(10), hists[idx].GetBinError(10), hists[idx].GetBinError(10)*xsections[idx]/numEventsList[idx])
    hists[idx].Scale(xsections[idx]/numEventsList[idx])
    print(hists[idx].GetBinError(10))
    hs.Add(hists[idx])

# to fit the hs we need to define a th1f that combines the counts of all the th1f
combined_hist = hists[0].Clone("combined_hist")
for hist in hists[1:]:
    combined_hist.Add(hist)
 
fitFunction = ROOT.TF1("DoubleSidedCrystalballFunction", DoubleSidedCrystalballFunction, 0, 350, 7)
fitFunction.SetNpx(5000)
fitFunction.SetParameters(0.1, 0.1, 0.1, 0.1, 90, 15, 10)
fitFunction.SetParLimits(0, 0, 10)
fitFunction.SetParLimits(1, 0, 10)
fitFunction.SetParLimits(2, 0, 20)
fitFunction.SetParLimits(3, 0, 10)
fitFunction.SetParLimits(4, 85, 95)
fitFunction.SetParLimits(5, 10, 30)
fitFunction.SetParLimits(6, 0, 20)
fitFunction.SetParNames("#alpha_l", "#alpha_r", "n_l", "n_r", "#mu", "#sigma", "N")

# Fit the combined histogram
print("Fit started")
xmaxFit = int(sys.argv[1]) if len(sys.argv)>1 else 300

#combined_hist.Draw("hist same")
combined_hist.Draw("hist")
combined_hist.Fit(fitFunction, "RE+", "", xmin, xmaxFit)
#st = combined_hist.Get("stats")
#st.Draw("same")



legend = CMS.cmsLeg(0.7, 0.3, 0.9, 0.5)
for idx, label in enumerate(labels):
    legend.AddEntry(hists[idx], label, "f")


combined_hist.Draw("hist same")
combined_hist.GetXaxis().SetTitle("Dijet Mass [GeV]")
combined_hist.GetYaxis().SetTitle("Counts [a.u.]")
hs.Draw("hist same")
fitFunction.SetLineColor(ROOT.kRed)
fitFunction.Draw("same")
legend.Draw()



#c1.Draw()
c1.SaveAs("/t3home/gcelotto/ggHbb/outputs/plots/fits/Zpeak_xmax%d.png"%xmaxFit)
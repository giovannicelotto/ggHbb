
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import loadMultiParquet, cut
import mplhep as hep
hep.style.use("CMS")
from array import array
import ROOT
from ROOT import RooFit, RooRealVar, RooDataSet, RooPlot, RooCrystalBall, RooArgList

# Step 1: Define the observable and parameters
x = RooRealVar("x", "Observable", 0, 350)
x0 = RooRealVar("x0", "Mean", 0, 80, 100)
sigmaL = RooRealVar("sigmaL", "Sigma Left", 1, 10, 30)
sigmaR = RooRealVar("sigmaR", "Sigma Right", 1, 10, 30)
alphaL = RooRealVar("alphaL", "Alpha Left", 2, 0.1, 30)
nL = RooRealVar("nL", "n Left", 2, 0.1, 30)
alphaR = RooRealVar("alphaR", "Alpha Right", 2, 0.1, 10)
nR = RooRealVar("nR", "n Right", 2, 0.1, 10)
crystalBall = RooCrystalBall("crystalBall", "Crystal Ball PDF", x, x0, sigmaL, sigmaR, alphaL, nL, alphaR, nR)
crystalBallTF1 = crystalBall.asTF(RooArgList(x))

def combined_fit(x, params):
    # Crystal Ball parameters
    alpha = params[0]
    n = params[1]
    mean = params[2]
    sigma = params[3]
    normCB = params[4]
    
    # Bernstein polynomial parameters
    #p0 = params[5]
    #p1 = params[6]
    #p2 = params[7]
    
    # One-sided Crystal Ball function
    crystal_ball = ROOT.Math.crystalball_function(x[0], alpha, n, mean, sigma)*normCB
    
    # Second-order Bernstein polynomial
    #bernstein_poly = p0 * (1 - x[0])**2 + 2 * p1 * x[0] * (1 - x[0]) + p2 * x[0]**2
    #pol = p0  + p1 * x[0] + p2*x[0]*x[0]
    
    return crystal_ball #+ pol

paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJets",
            ]

xsections = [5.261e+03,
 1012, 114.2, 25.34, 12.99, 9.8, 
 ]
labels = [
'ZJets100-200', 'ZJets200-400', 'ZJets400-600', 'ZJets600-800', 'ZJets800-Inf',
'EWKZJets']

dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=-2, nMC=-1, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt'], returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)

dfs = cut(dfs, 'jet1_pt', 20, None)
dfs = cut(dfs, 'jet2_pt', 20, None)

fig, ax = plt.subplots(1, 1)
xmin, xmax = 20, 350
bins = np.linspace(xmin, xmax, 40)
for idx, df in enumerate(dfs):
    counts = np.histogram(df.dijet_mass, bins=bins, weights=df.sf*xsections[idx]/numEventsList[idx])[0]
    ax.hist(bins[:-1], bins=bins, weights=counts, label = labels[idx])
hep.cms.label()
ax.legend()
ax.set_xlabel("Dijet mass [GeV]")
outName = "/t3home/gcelotto/ggHbb/outputs/plots/fits/zpeak.png"
fig.savefig(outName)
print("saved in", outName)


ROOT.gStyle.SetOptStat(0000)
ROOT.gStyle.SetOptFit(1111)


c1 = ROOT.TCanvas("c1", "c1", 800, 600)
hist100 = ROOT.TH1F("hist100", "", len(bins) - 1, bins)
hist200 = ROOT.TH1F("hist200", "", len(bins) - 1, bins)
hist400 = ROOT.TH1F("hist400", "", len(bins) - 1, bins)
hist600 = ROOT.TH1F("hist600", "", len(bins) - 1, bins)
hist800 = ROOT.TH1F("hist800", "", len(bins) - 1, bins)
histEWK = ROOT.TH1F("histEWK", "", len(bins) - 1, bins)
hists = [
hist100 ,hist200 ,hist400 ,hist600 ,hist800,
histEWK
]
colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kOrange, ROOT.kMagenta, ROOT.kYellow]
hs = ROOT.THStack("hs", "Stacked Histograms")
print("For started")
for idx, df in enumerate(dfs):
    counts = np.histogram(df.dijet_mass, bins=bins, weights=df.sf)[0]
    for i in range(len(counts)):
        hists[idx].SetBinContent(i + 1, counts[i])
    hists[idx].SetFillColor(colors[idx])
    hists[idx].Scale(xsections[idx]/numEventsList[idx])
    hs.Add(hists[idx])

# to fit the hs we need to define a th1f that combines the counts of all the th1f
combined_hist = hists[0].Clone("combined_hist")
for hist in hists[1:]:
    combined_hist.Add(hist)
#for h in hists:
#    print(h.GetBinError(10))
#print(combined_hist.GetBinError(10))

xmin, xmax = 20, 350
#fit_bkg = ROOT.TF1("fit_bkg", "[0] + [1]*x + [2]*x*x", xmin, xmax)
#fit_bkg.SetParameters(0.1,0.01, 0.001)
#combined_hist.Fit(fit_bkg, "RE","",180, xmax)
 
# Create the TF1 object
fit_function = ROOT.TF1("fit_function", combined_fit, xmin, xmax, 5)

# Set initial parameter values for the fit (example values, may need adjustment)
#fit_function.SetParameters(90, 15, 1.0, 4, 0.5, 0.5, 0.5, 0.5)
fit_function.SetParameters(1., 4, 15, 90, 6)#, fit_bkg.GetParameter(0), fit_bkg.GetParameter(1), fit_bkg.GetParameter(2))
fit_function.SetParLimits(0, 1, 10)
fit_function.SetParLimits(1, 0, 40)
fit_function.SetParLimits(2, 10, 20)
fit_function.SetParLimits(3, 85, 95)
fit_function.SetParLimits(4, 4, 10)


# Fit the combined histogram
print("Fit started")
fit_function.SetParNames(r"#alpha", "n",r"#sigma",r"#mu","Norm")#, "p0","p1","p2")
combined_hist.Fit(crystalBallTF1, "RE+","",xmin, xmax)

c1.Update()
#pol2 = ROOT.TF1("pol2", "[0] + [1]*x + [2]*x*x", xmin, xmax)
#pol2.SetParameters(fit_function.GetParameter(5), fit_function.GetParameter(6), fit_function.GetParameter(7))
#pol2.SetLineColor(ROOT.kGreen)


# Draw the fit on the combined histogram
#fit_function.Draw("same")

legend = ROOT.TLegend(0.7, 0.3, 0.9, 0.5)
for idx, label in enumerate(labels):
    legend.AddEntry(hists[idx], label, "f")


combined_hist.Draw("hist")
hs.Draw("hist same")
#pol2.Draw("same")
fit_function.SetLineColor(ROOT.kBlack)
fit_function.Draw("same")
legend.Draw()


#c1.Update()
c1.Draw()
c1.SaveAs("/t3home/gcelotto/ggHbb/outputs/plots/fits/stacked_histograms.png")
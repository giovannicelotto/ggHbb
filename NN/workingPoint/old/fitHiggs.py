# %%
import glob, re, sys
sys.path.append('/t3home/gcelotto/ggHbb/NN')
import numpy as np
import pandas as pd
from functions import loadMultiParquet, cut, getXSectionBR
from helpersForNN import preprocessMultiClass, scale, unscale
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import mplhep as hep
hep.style.use("CMS")
import math
from applyMultiClass_Hpeak import getPredictions, splitPtFunc
from ROOT import TCanvas, TH1F
# %%
#
# load Data
#

from loadData import loadData
dfs, YPred_H = loadData()

# %%
xmin = 70
xmax = 200

workingPoint = (YPred_H[:,1]>0.34) & (YPred_H[:,0]<0.22)

fig,  ax = plt.subplots(1, 1)
bins=np.linspace(xmin, xmax, 100)
counts = np.histogram(dfs[0].dijet_mass[workingPoint], bins=bins, weights=dfs[0].sf[workingPoint])[0]
ax.hist(bins[:-1], bins=bins, weights=counts)
ax.set_xlim(xmin, xmax)
# %%


canvas = TCanvas("canvas", "Histogram", 800, 600)
hist = TH1F("hist", ";X-axis Title;Y-axis Title", len(bins) - 1, bins)

for mass, weight in zip(np.array(dfs[0].dijet_mass)[workingPoint], np.array(dfs[0].sf)[workingPoint]):
    hist.Fill(mass, weight)

hist.Draw("HIST")
hist.GetXaxis().SetRangeUser(xmin, xmax)

canvas.Update()
canvas.SaveAs("histogram.png")


# %%
import ROOT
from fitFunction import DoubleSidedCrystalballFunction

ROOT.gStyle.SetOptStat(0000)
ROOT.gStyle.SetOptFit(1111)
fitFunction = ROOT.TF1("fitFunction", DoubleSidedCrystalballFunction, xmin, xmax, 7)
fitFunction.SetNpx(10000)
fitFunction.SetParameters(0.5, 2, 1, 1, 125, 15, 9000,30, -0.03, 0.0001)
fitFunction.SetParLimits(0, 0, 5)
fitFunction.SetParLimits(1, 0, 5)
#fitFunction.SetParLimits(2, 0, 10)
fitFunction.SetParLimits(3, 0, 40)
fitFunction.SetParLimits(4, 110, 145)
fitFunction.SetParLimits(5, 10, 30)
fitFunction.SetParLimits(6, 5e3, 20e3)
#fitFunction.SetParLimits(8, -1, 1)
fitFunction.SetParNames("#alpha_l", "#alpha_r", "n_l", "n_r", "#mu", "#sigma", "N")
hist.Fit(fitFunction, "RE+","", xmin, xmax)


legend = ROOT.TLegend(0.15, 0.7, 0.35, 0.8)
legend.AddEntry(hist, 'GluGluHToBB', "f")

hist.GetXaxis().SetTitle("Dijet Mass [GeV]")
hist.GetYaxis().SetTitle("Counts [a.u.]")
hist.Draw("histe")


fitFunction.SetLineColor(ROOT.kRed)
fitFunction.Draw("same")
legend.Draw()
gaus = ROOT.TF1("gaus", "gaus(0)", xmin, xmax, 3)
gaus.SetParameters(fitFunction.GetParameter(6), fitFunction.GetParameter(4), fitFunction.GetParameter(5))
gaus.SetLineColor(ROOT.kGray)
gaus.Draw("same")

#c1.Update()
canvas.Draw()
canvas.SaveAs("/t3home/gcelotto/ggHbb/NN/workingPoint/higgs.png")
# %%
root_file = ROOT.TFile("/t3home/gcelotto/ggHbb/NN/workingPoint/tree.root", "RECREATE")
hist.Write()
from array import array
mass                 = array('d', [0]) 
sf                 = array('d', [0]) 
tree = ROOT.TTree("tree", "Tree")
tree.Branch("mass", mass,      "mass/D")  
tree.Branch("sf",   sf,          "sf/D")  
for mass_, weight in zip(np.array(dfs[0].dijet_mass)[workingPoint], np.array(dfs[0].sf)[workingPoint]):

    mass[0]      = mass_
    sf[0]      = weight
    tree.Fill()

# %%
    
# %%
fitFunction.Write()
tree.Write()
# %%


root_file.Close()
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
# %%
pTClass, nReal, nMC = 0, -2, -1
paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others",
            ]

pathToPredictions = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions"
# check for which fileNumbers the predictions is available
isMCList = [1]
fileNumberList = []
for isMC in isMCList:
    fileNumberProcess = []
    fileNamesProcess = glob.glob(pathToPredictions+"/yMC%d_fn*pt%d*.parquet"%(isMC, pTClass))
    for fileName in fileNamesProcess:
        match = re.search(r'_fn(\d+)_pt', fileName)
        if match:
            fn = match.group(1)
            fileNumberProcess.append(int(fn))
            
        else:
            pass
            #print("Number not found")
    fileNumberList.append(fileNumberProcess)
    print(len(fileNumberProcess), " predictions files for process MC : ", isMC)

# %%
# load the files where the prediction is available
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'jet1_qgl', 'jet2_qgl', 'dijet_dR', 'dijet_dPhi'], returnNumEventsTotal=True, selectFileNumberList=fileNumberList, returnFileNumberList=True)
pTmin, pTmax, suffix = [[0,-1,'inclusive'], [0, 30, 'lowPt'], [30, 100, 'mediumPt'], [100, -1, 'highPt']][pTClass]    
for df in dfs:
    print()
dfs = preprocessMultiClass(dfs, pTmin, pTmax, suffix)   # get the dfs with the cut in the pt class

minPt, maxPt = None, None #180, -1
if (minPt is not None) | (maxPt is not None):
    dfs, masks = splitPtFunc(dfs, minPt, maxPt)
    splitPt = True
else:
    masks=None
    splitPt=False
        

W_H = dfs[0].sf*getXSectionBR()/numEventsList[0]

YPred_H= getPredictions(fileNumberList, pathToPredictions, splitPt=splitPt, masks=masks, isMC=isMCList, pTClass=pTClass)
YPred_H = np.array(YPred_H)[0]
# %%
fig,  ax = plt.subplots(1, 1)
bins=np.linspace(40, 200, 100)

workingPoint = (YPred_H[:,1]>0.34) & (YPred_H[:,0]<0.22)

counts = np.histogram(dfs[0].dijet_mass[workingPoint], bins=bins, weights=dfs[0].sf[workingPoint])[0]
ax.hist(bins[:-1], bins=bins, weights=counts)
ax.set_xlim(40, 200)
# %%
from ROOT import TCanvas, TH1F
canvas = TCanvas("canvas", "Histogram", 800, 600)
hist = TH1F("hist", ";X-axis Title;Y-axis Title", len(bins) - 1, bins)

for mass, weight in zip(np.array(dfs[0].dijet_mass)[workingPoint], np.array(dfs[0].sf)[workingPoint]):
    hist.Fill(mass, weight)

# Draw the histogram
hist.Draw("HIST")
hist.GetXaxis().SetRangeUser(40, 200)

# Update the canvas to display the histogram
canvas.Update()
canvas.SaveAs("histogram.png")


# %%
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

def DoubleSidedCrystalballFunctionPlusBackground(x, par):
    alpha_l = par[0]
    alpha_r = par[1]
    n_l = par[2]
    n_r = par[3]
    mean = par[4]
    sigma = par[5]
    N = par[6]
    p0 =par[7]
    p1 = par[8]

    
    t = (x[0] - mean) / sigma
    result = 0.0
    
    fact1TLessMinosAlphaL = alpha_l / n_l
    fact2TLessMinosAlphaL = (n_l / alpha_l) - alpha_l - t
    
    fact1THihgerAlphaH = alpha_r / n_r
    fact2THigherAlphaH = (n_r / alpha_r) - alpha_r + t
    
    if -alpha_l <= t <= alpha_r:
        result = np.exp(-0.5 * t * t) + p0 + p1*x[0] 
    elif t < -alpha_l:
        result = np.exp(-0.5 * alpha_l * alpha_l) * np.power(fact1TLessMinosAlphaL * fact2TLessMinosAlphaL, -n_l) + p0 + p1*x[0] 
    elif t > alpha_r:
        result = np.exp(-0.5 * alpha_r * alpha_r) * np.power(fact1THihgerAlphaH * fact2THigherAlphaH, -n_r)+ p0 + p1*x[0] 
    
    return N * result

ROOT.gStyle.SetOptStat(0000)
ROOT.gStyle.SetOptFit(1111)
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
fitFunction.SetParNames("#alpha_l", "#alpha_r", "n_l", "n_r", "#mu", "#sigma", "N")
hist.Fit(fitFunction, "REN+","",40, 200)




fitFunctionPlusBackground = ROOT.TF1("fitFunctionPlusBackground", DoubleSidedCrystalballFunction, 40, 200, 9)
fitFunctionPlusBackground.SetNpx(5000)
#for i in range(7):
#    fitFunctionPlusBackground.SetParameter(i, fitFunction.GetParameter(i))
fitFunctionPlusBackground.SetParameter(0, fitFunction.GetParameter(0))
fitFunctionPlusBackground.SetParameter(1, fitFunction.GetParameter(1))
fitFunctionPlusBackground.SetParameter(2, 130)
fitFunctionPlusBackground.SetParameter(3, fitFunction.GetParameter(3))
fitFunctionPlusBackground.SetParameter(4, fitFunction.GetParameter(4))
fitFunctionPlusBackground.SetParameter(5, fitFunction.GetParameter(5))
fitFunctionPlusBackground.SetParameter(6, fitFunction.GetParameter(6))
fitFunctionPlusBackground.SetParameter(7,200)
fitFunctionPlusBackground.SetParameter(8,-0.2)
fitFunctionPlusBackground.SetParameter(9,0.001)
fitFunctionPlusBackground.SetParLimits(0, 0, 5)
fitFunctionPlusBackground.SetParLimits(1, 0, 5)
fitFunctionPlusBackground.SetParLimits(2, 0, 10)
fitFunctionPlusBackground.SetParLimits(3, 0, 40)
fitFunctionPlusBackground.SetParLimits(4, 110, 145)
fitFunctionPlusBackground.SetParLimits(5, 10, 30)
fitFunctionPlusBackground.SetParLimits(6, 0, 100e3)
fitFunctionPlusBackground.SetParLimits(7, 0, 500)
fitFunctionPlusBackground.SetParLimits(8, -1, 1)
#fitFunctionPlusBackground.SetParLimits(9, -1, 1)
fitFunctionPlusBackground.SetParNames("#alpha_l", "#alpha_r", "n_l", "n_r", "#mu", "#sigma", "N", "p0", "p1")

hist.Fit(fitFunctionPlusBackground, "RE+","",40, 200)


legend = ROOT.TLegend(0.15, 0.6, 0.35, 0.8)
legend.AddEntry(hist, 'GluGluHToBB', "f")
legend.AddEntry(fitFunctionPlusBackground, 'Fit Function', "l")

hist.GetXaxis().SetTitle("Dijet Mass [GeV]")
hist.GetYaxis().SetTitle("Counts [a.u.]")
hist.Draw("hist")


#pol2.Draw("same")
fitFunctionPlusBackground.SetLineColor(ROOT.kRed)
fitFunctionPlusBackground.Draw("same")
legend.Draw()
gaus = ROOT.TF1("gaus", "gaus(0)", 40, 300, 3)
gaus.SetParameters(fitFunctionPlusBackground.GetParameter(6), fitFunctionPlusBackground.GetParameter(4), fitFunctionPlusBackground.GetParameter(5))
gaus.SetLineColor(ROOT.kGray)
gaus.Draw("same")

pol2 = ROOT.TF1("pol2", "pol2(0)", 40, 200, 3)
pol2.SetParameters(fitFunctionPlusBackground.GetParameter(7), fitFunctionPlusBackground.GetParameter(8), fitFunctionPlusBackground.GetParameter(9))
pol2.SetLineColor(ROOT.kBlue)
pol2.Draw("same")


#c1.Update()
canvas.Draw()
canvas.SaveAs("/t3home/gcelotto/ggHbb/NN/workingPoint/higgs.png")
# %%

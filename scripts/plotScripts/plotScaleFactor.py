import matplotlib.pyplot as plt
import numpy as np
import ROOT
from utilsForPlot import getBins, loadData, loadDataOnlyFeatures
import mplhep as hep
hep.style.use("CMS")

def plotSF():
    histPath = "/t3home/gcelotto/ggHbb/trgMu_scale_factors.root"
    f = ROOT.TFile(histPath, "READ")
    hist = f.Get("hist_scale_factor")
    toKeep = [29, 33]
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/muonIso"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/muonIso"
    signal, realData = loadDataOnlyFeatures(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=-1, features=toKeep)
    signalWeight = np.ones(len(signal))
    for i in range(len(signal)):
        signalWeight[i]= hist.GetBinContent(hist.GetXaxis().FindBin(signal[i,0]),hist.GetYaxis().FindBin(abs(signal[i,1])))

    fig, ax = plt.subplots(1, 1)
    bins=np.linspace(0, 80, 100)
    ax.hist(signal[:,0], bins=bins, histtype=u'step', color='blue', label='Muon pT Unweighted')
    ax.hist(signal[:,0], bins=bins, weights=signalWeight, histtype=u'step', color='red',  label='Muon pT Rescaled')
    ax.set_xlabel("Muon p$_{T}$ [GeV]")
    ax.set_ylabel("Events")
    ax.legend()
    fig.savefig("/t3home/gcelotto/ggHbb/outputs/muon_pt_rescaled.png", bbox_inches='tight')

    f.Close()


if __name__=="__main__":
    plotSF()



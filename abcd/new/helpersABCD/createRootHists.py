import ROOT
import numpy as np
def createRootHists(countsDict, hB_ADC, regions, bins, suffix):
    
    # Create a ROOT file to store histograms
    root_file = ROOT.TFile("/t3home/gcelotto/ggHbb/abcd/combineTry/shapes/counts_%s.root"%(suffix), "RECREATE")
    
    # Create histograms for each process
    # use same bins
    processes = ['H', 'VV', 'ST', 'ttbar', 'W+Jets', 'Z+Jets', 'QCD']
    hists = {
        'H':      countsDict["H"]  ,
        'VV':     countsDict["VV"]  ,
        'ST':     countsDict["ST"]  ,
        'ttbar':  countsDict["ttbar"]      ,
        'W+Jets': countsDict["W+Jets"]      ,
        'Z+Jets': countsDict["Z+Jets"]      ,
        'QCD':    hB_ADC,
        'data_obs':    regions["B"]
    }
    
    # Create histograms for each process
    for proc, hist in hists.items():
        rootHist = ROOT.TH1F(proc, proc, len(bins)-1, bins)
        for i, (value, error) in enumerate(zip(hist.values(), np.sqrt(hist.variances()))):
            rootHist.SetBinContent(i+1, value)
            rootHist.SetBinError(i+1, error)
        rootHist.Write()
    
    # Close the file
    root_file.Close()
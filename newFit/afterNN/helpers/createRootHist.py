import ROOT
import numpy as np
def createRootHists(hists, labels, bins, outFolder, suffix):
    '''
    hists : list of Hist object with values and variances
    labels : list of str with of corresponding hist
    bins : bins of the hist
    suffx : str to characterize the saved hist
    '''
    
    outName=outFolder+"/counts_%s.root"%(suffix)
    root_file = ROOT.TFile(outName, "RECREATE")
    
    for hist, label in zip(hists, labels):
        rootHist = ROOT.TH1F(label, label, len(bins)-1, bins)
        for i, (value, error) in enumerate(zip(hist.values(), np.sqrt(hist.variances()))):
            rootHist.SetBinContent(i+1, value)
            rootHist.SetBinError(i+1, error)
        rootHist.Write()
    
    root_file.Close()
    print("Closed ROOT file %s"%outName)
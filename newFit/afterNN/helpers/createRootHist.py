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




def createRootHists_Syst(hists, labels, bins, outFolder, suffix, systematics=None):
    '''
    hists : list of Hist objects with values and variances (nominal histograms)
    labels : list of str corresponding to each nominal histogram
    bins : array-like, bin edges for the histograms
    outFolder : str, path to the output directory
    suffix : str, descriptor for the saved histograms
    systematics : dict (optional), maps labels to their systematic variations
                  e.g., {'signal': {'up': hist_up, 'down': hist_down}, ...}
    '''
    
    outName = f"{outFolder}/counts_{suffix}.root"
    root_file = ROOT.TFile(outName, "RECREATE")

    # Save nominal histograms
    for hist, label in zip(hists, labels):
        rootHist = ROOT.TH1F(label, label, len(bins)-1, bins)
        for i, (value, error) in enumerate(zip(hist.values(), np.sqrt(hist.variances()))):
            rootHist.SetBinContent(i+1, value)
            rootHist.SetBinError(i+1, error)
        rootHist.Write()

        # If systematics are provided, save them
        if systematics and label in systematics:
            for var in ['up', 'down']:
                if var in systematics[label]:
                    syst_hist = systematics[label][var]
                    syst_label = f"{label}_{var.capitalize()}"  # e.g., "signal_Up"
                    rootSystHist = ROOT.TH1F(syst_label, syst_label, len(bins)-1, bins)
                    for i, (value, error) in enumerate(zip(syst_hist.values(), np.sqrt(syst_hist.variances()))):
                        rootSystHist.SetBinContent(i+1, value)
                        rootSystHist.SetBinError(i+1, error)
                    rootSystHist.Write()
    
    root_file.Close()
    print(f"Closed ROOT file {outName}")

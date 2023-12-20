import numpy as np
import uproot
import matplotlib.pyplot as plt
import glob
import sys
import mplhep as hep
hep.style.use("CMS")

def main():
    cTagSignal = []
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggHcc2023Dec14/GluGluHToCC_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8/crab_GluGluHToCC/231214_162727/0000" 
    signalFiles = glob.glob(signalPath+"/HccAlone.root")
    print("List of files %d"%len(signalFiles))
    for fileName in signalFiles[:10]:
        sys.stdout.write('\r')
        sys.stdout.write("%d/%d"%(signalFiles.index(fileName)+1, len(signalFiles)))
        sys.stdout.flush()
        f=uproot.open(fileName)
        tree=f['Events']
        branches = tree.arrays()
        maxEntries = tree.num_entries
        deepFlavC = branches['Jet_btagDeepFlavC']
        
        cTagSignal+=list(np.max(deepFlavC, axis=1))
        #cTagSignal+=list(deepFlavC[:,0])

    
    dataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30/ParkingBPH1/crab_data_Run2018A_part1/231130_120505/0000"
    cTagData = []
    dataFiles = glob.glob(dataPath+"/Data*.root")
    print("List of files %d"%len(dataFiles))
    for fileName in dataFiles[:1]:
        f=uproot.open(fileName)
        tree=f['Events']
        branches = tree.arrays()
        maxEntries = tree.num_entries
        print(maxEntries)
        deepFlavC = branches['Jet_btagDeepFlavC']
        
        toAppend = np.max(deepFlavC, axis=1)
        print(len(toAppend))
        cTagData+=list(toAppend)
        #cTagData+=list(deepFlavC[:,0])

    
    fig, ax = plt.subplots(1, 1)
    bins = np.linspace(0, 1, 201)
    cs = np.histogram(cTagSignal, bins=bins)[0]
    cb = np.histogram(cTagData, bins=bins)[0]
    cs=cs/np.sum(cs)
    cb=cb/np.sum(cb)
    ax.hist(bins[:-1], bins=bins, weights=cs, color='blue', histtype=u'step', label='signal')
    ax.hist(bins[:-1], bins=bins, weights=cb, color='red', histtype=u'step', label='data')
    #ax.set_xlabel("Leading Jet_btagDeepFlavourC")
    ax.set_xlabel("Max Jet_btagDeepFlavourC")
    ax.set_ylabel("Events")
    outName = "/t3home/gcelotto/ggHbb/outputs/plots/cFlavorMax.png"
    #outName = "/t3home/gcelotto/ggHbb/outputs/plots/cFlavorLeading.png"
    print("saving in ", outName)
    fig.savefig(outName, bbox_inches='tight')

    thresholds= np.linspace(0, 1, 101)
    cTagData = np.array(cTagData)
    cTagSignal = np.array(cTagSignal)
    retainedSignal, retainedData = [], []
    for t in thresholds:
        retainedSignal.append(np.sum(cTagSignal>t)/len(cTagSignal))
        retainedData.append(np.sum(cTagData>t)/len(cTagData))
    fig, ax =plt.subplots(1, 1)
    ax.plot(np.ones(len(retainedData))-retainedData, retainedSignal, color='blue')
    ax.set_xlabel("Background Rejection [%]")
    ax.set_ylabel("Signal Efficiency [%]")
    #ax.plot(thresholds, retainedData, color='red')
    outName = "/t3home/gcelotto/ggHbb/outputs/plots/rocCFlavor.png"
    fig.savefig(outName, bbox_inches='tight')
    print("saving in ", outName)

    fig, ax =plt.subplots(1, 1)
    ax.plot(thresholds, retainedSignal, color='blue', label='signal')
    ax.plot(thresholds, retainedData, color='red', label='Data')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Efficiency [%]")
    outName = "/t3home/gcelotto/ggHbb/outputs/plots/EffVsThreshold.png"
    fig.savefig(outName, bbox_inches='tight')
    print("saving in ", outName)




    return 0


if __name__=="__main__":
    main()
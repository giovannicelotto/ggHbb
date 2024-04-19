import matplotlib.pyplot as plt
import numpy as np
import uproot
import mplhep as hep
import awkward as ak
hep.style.use("CMS")

def main():
    path_preUL   = "/t3home/gcelotto/CMSSW_12_4_8/src/PhysicsTools/BParkingNano/test/ggHbb_preUL_Run2_mc_124X.root"
    path_UL      = "/t3home/gcelotto/CMSSW_12_4_8/src/PhysicsTools/BParkingNano/test/ggHbb_UL_Run2_mc_124X.root"
    f_preUL = uproot.open(path_preUL)
    f_UL    = uproot.open(path_UL)
    tree_preUL  = f_preUL['Events']
    tree_UL     = f_UL['Events']
    branches_preUL  = tree_preUL.arrays()
    branches_UL     = tree_UL.arrays()
    Jet_pt_preUL    = branches_preUL['Jet_pt']        
    Jet_pt_UL       = branches_UL['Jet_pt']
    bins=np.linspace(0, 50, 100)
    fig, ax = plt.subplots(1, 1)
    ax.hist(ak.flatten(Jet_pt_preUL), bins=bins, color='blue', label='PRE UL', histtype=u'step', linewidth=2)
    ax.hist(ak.flatten(Jet_pt_UL), bins=bins, color='red', label='UL', histtype=u'step')
    ax.legend()
    outName = "/t3home/gcelotto/ggHbb/outputs/plots/others/preULvsUL_fromCMSSW.png"
    fig.savefig(outName, bbox_inches='tight')
    print("Saving in ", outName)

    return


if __name__ =="__main__":
    main()
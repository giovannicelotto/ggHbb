import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import glob
hep.style.use("CMS")
def main():
    path="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH_2023Nov30/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231130_120412/genOnly"
    fileNames = glob.glob(path+"/*.npy")
    
    for fileName in fileNames[:]:
        try:
            f = np.load(fileName)
            signal = np.concatenate(signal, f)
        except:
            signal=np.load(fileName)
        
    
    
    fig, ax = plt.subplots(1, 1)
    bins= np.arange(15)
    c0 = np.histogram(signal[:,0], bins=bins)[0]
    c0=c0/np.sum(c0)
    c1 = np.histogram(signal[:,1], bins=bins)[0]
    c1=c1/np.sum(c1)
    
    ax.hist(bins[:-1], weights=c0, bins=bins, color='blue', label='Leading Daughter')
    print(c0.shape, c1.shape, bins.shape)
    ax.hist(bins[:-1], weights=c1, bins=bins, color='purple', bottom=c0, label='Subleading Daughter')
    outName = "/t3home/gcelotto/ggHbb/outputs/plots/genmatched_jetIdx.png"
    print("Saving in %s"%outName)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Jet Index (ranked by p$_\mathrm{T}$)")
    ax.set_ylabel("Probability to be daughter of Higgs")
    ax.legend(loc='upper right')
    for i in range(len(c0)):
        ax.text(bins[i], y=c0[i]+c1[i]+0.025, s=str(round(c0[i]*100, 1))+"%", fontsize=14, color='blue')
        ax.text(bins[i], y=c0[i]+c1[i]+0.05, s=str(round(c1[i]*100, 1))+"%", fontsize=14, color='purple')
    fig.savefig(outName, bbox_inches='tight')


if __name__=="__main__":
    main()

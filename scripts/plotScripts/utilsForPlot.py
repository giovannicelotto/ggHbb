import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

def getFeaturesBScoreBased(number=False):
    featureNames = [
        'Jet1Pt', 'Jet1Eta', 'Jet1Phi', 'Jet1Mass', 
        'Jet1_nMuons', 'Jet1_nElectrons', 'Jet1_btagDeepFlavB', 'Jet1_Area', 'Jet1_QGl',
        'Jet2Pt', 'Jet2Eta', 'Jet2Phi', 'Jet2Mass',
        'Jet2_nMuons', 'Jet2_nElectrons', 'Jet2_btagDeepFlavB', 'Jet2_Area', 'Jet2_QGl',
        'DijetPt', 'DijetEta', 'DijetPhi', 'DijetMass',
        'dR_Dijet', 'dEta_Dijet', 'dPhi_Dijet',  'd_Eta+Pi-dPhi',
        'twist_Dijet', 'ht'
        ]
    if number:
        for i in range(len(featureNames)):
            featureNames[i]=str(i)+"_"+featureNames[i]
            
    return featureNames.copy()


def getBins():
    bins=np.array([
        [0,  200,], [-4,  4,],    [-np.pi, np.pi,], [0,  30,],
        [0, 10],    [0, 6],       [0, 1],   [0.3, 0.7], [-0.5, 1],
        [0,  120,], [-4,  4,],    [-np.pi, np.pi,], [0,  30,],
        [0, 10],    [0, 5],       [0, 1], [0.3, 0.7], [-0.5, 1],
        [0, 300],   [-5, 5],        [-np.pi, np.pi], [0, 450],
        [0, 6],     [0, 6],     [0, np.pi],     [0, 10],
        [0, np.pi/2], [0, 800]
        
        ])
    
    return bins.copy()


def scatter2Features(sig, bkg, labels,bins, outFile, figsize=0):
    ''' function that makes a scatter plot with all the variables given as argument for signal and background'''
    if figsize==0:
        figsize=sig.shape[1]*3
    numFeatures = sig.shape[1]
    fig, ax = plt.subplots(numFeatures, numFeatures, figsize=(figsize, figsize), constrained_layout=True)
    for i in range(numFeatures):
        
        for j in range(i+1, numFeatures):
            ax[i, j].scatter(bkg[:,j], bkg[:,i], alpha=0.5, s=2, color='red')
            ax[i, j].scatter(sig[:,j], sig[:,i], alpha=0.5, s=2, color='blue')
            ax[i, j].set_xlim(bins[j])
            ax[i, j].set_ylim(bins[i])

    for i in range(numFeatures):
            c, b_ = ax[i, i].hist(bkg[bkg[:,i]>-998,i], bins=30, range=bins[i], label='Background', histtype=u'step' , color='red', density=True)[:2]
            ax[i, i].hist(sig[sig[:,i]>-998,i], bins=b_, label='Signal', histtype=u'step',  color='blue', density=True)[:2]
            ax[i, i].set_ylabel(labels[i], fontsize=20)
            ax[i, i].set_xlim(bins[i])
            ax[i, i].legend()
            ax[i, i].set_xlabel(labels[i], fontsize=20)
            ax[i, i].set_yscale('log')
        
    

    for i in range(numFeatures):
        for j in range(numFeatures):
            if j<i:
                ax[i, j].set_visible(False)
    print("SAVING")
    fig.savefig(outFile, bbox_inches='tight')


def loadData(signalPath, realDataPath, nSignalFiles, nRealDataFiles):
    print("Loading Data...")
    signalFileNames = glob.glob(signalPath+"/*bScoreBased4_*.npy")
    realDataFileNames = glob.glob(realDataPath+"/*bScoreBased4_*.npy")
    signalFileNames = signalFileNames[:nSignalFiles] if nSignalFiles!=-1 else signalFileNames
    realDataFileNames = realDataFileNames[:nRealDataFiles] if nRealDataFiles!=-1 else realDataFileNames

    print("%d files for MC ggHbb" %len(signalFileNames))
    print("%d files for realDataFileNames" %len(realDataFileNames))
    signal = np.load(signalFileNames[0])[:,:]
    for signalFileName in signalFileNames[1:]:
        sys.stdout.write('\r')
        sys.stdout.write("   %d/%d   "%(signalFileNames.index(signalFileName)+1, len(signalFileNames)))
        sys.stdout.flush()

        currentSignal = np.load(signalFileName)[:,:]
        signal = np.concatenate((signal, currentSignal))
    print("Signal shape: ", signal.shape)

    bscore4 = np.load(realDataFileNames[0])[:,:]
    for bscore4FileName in realDataFileNames[1:]:
        sys.stdout.write('\r')
        sys.stdout.write("   %d/%d   "%(realDataFileNames.index(bscore4FileName)+1, len(realDataFileNames)))
        sys.stdout.flush()
        currentBscore4 = np.load(bscore4FileName)[:,:]
        bscore4 = np.concatenate((bscore4, currentBscore4))
    print("bscore4 shape: ", bscore4.shape)

    return signal, bscore4

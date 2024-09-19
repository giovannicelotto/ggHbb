import numpy as np
import matplotlib.pyplot as plt
import sys
import os, re
import glob
import pandas as pd
#import dask.dataframe as dd
def loadParquet(signalPath, realDataPath, nSignalFiles=-1, nRealDataFiles=1, columns=None, returnNumEventsTotal=False):

    signalFileNames = glob.glob(signalPath+"/*.parquet", recursive=True)
    realDataFileNames = glob.glob(realDataPath+"/*.parquet", recursive=True)
    signalFileNames = signalFileNames[:nSignalFiles] if nSignalFiles!=-1 else signalFileNames
    realDataFileNames = realDataFileNames[:nRealDataFiles] if nRealDataFiles!=-1 else realDataFileNames

    print("%d files for MC ggHbb" %len(signalFileNames))
    print("%d files for realDataFileNames" %len(realDataFileNames))
    
    signal = pd.read_parquet(signalFileNames, columns=columns)
    realData = pd.read_parquet(realDataFileNames, columns=columns)
    if returnNumEventsTotal:
        numEventsTotal=0
        df = pd.read_csv("/t3home/gcelotto/ggHbb/abcd/output/miniDf.csv")
        for fileName in signalFileNames:
            filename = os.path.splitext(os.path.basename(fileName))[0]
            process = filename.split('_')[0]  # split the process and the fileNumber and keep the process only which is GluGluHToBB in this case
            fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))
            numEventsTotal = numEventsTotal + df[(df.process==process) & (df.fileNumber==fileNumber)].numEventsTotal.iloc[0]
        return signal, realData, numEventsTotal
    else:
        return signal, realData

def loadDask(signalPath, realDataPath, nSignalFiles, nRealDataFiles, columns):
    signalFileNames = glob.glob(signalPath+"/*.parquet")
    realDataFileNames = glob.glob(realDataPath+"/*.parquet")
    signalFileNames = signalFileNames[:nSignalFiles] if nSignalFiles!=-1 else signalFileNames
    realDataFileNames = realDataFileNames[:nRealDataFiles] if nRealDataFiles!=-1 else realDataFileNames    

    print("%d files for MC ggHbb" %len(signalFileNames))
    print("%d files for realDataFileNames" %len(realDataFileNames))

    
    try:    
        signal = dd.read_parquet(signalFileNames, columns=columns)
        realData = dd.read_parquet(realDataFileNames, columns=columns)
        return signal, realData
    except:
        print("Some of the files might be corrupted. Here is the list:\n")
        for fileName in signalFileNames:
            try:
                df=pd.read_parquet(fileName)
            except:
                print(fileName)
        for fileName in realDataFileNames:
            try:
                df=pd.read_parquet(fileName)
            except:
                print(fileName)
        sys.exit("Exiting the program due to a corrupted files.")


def loadRoot(signalPath, realDataPath, nSignalFiles=-1, nRealDataFiles=1):
    signalFileNames = glob.glob(signalPath)
    realDataFileNames = glob.glob(realDataPath)
    signalFileNames = signalFileNames[:nSignalFiles] if nSignalFiles!=-1 else signalFileNames
    realDataFileNames = realDataFileNames[:nRealDataFiles] if nRealDataFiles!=-1 else realDataFileNames

    signalChain = ROOT.TChain("Events")
    for fileName in signalFileNames[:]:
        sys.stdout.write('\r')
        sys.stdout.write("   %d/%d   "%(signalFileNames.index(fileName)+1,len(signalFileNames)))
        sys.stdout.flush()
        signalChain.Add(fileName)
    print("\nOpening the chain")
    rdf = ROOT.RDataFrame(signalChain)  
    branches = signalChain.GetListOfBranches()
    featureNames = [branch.GetName() for branch in branches]
    print("Numpy conversion")
    npy = rdf.AsNumpy(columns=featureNames) # get numpy arrays ordered
    signal = np.column_stack(list(npy.values()))
    
    
    print("\nOpening Real Data")
    realDataChain = ROOT.TChain("Events")
    for fileName in realDataFileNames[:1]:
        sys.stdout.write('\r')
        sys.stdout.write("   %d/%d   "%(realDataFileNames.index(fileName)+1,len(realDataFileNames)))
        sys.stdout.flush()
        realDataChain.Add(fileName)
    print("Opening the chain")
    rdf2 = ROOT.RDataFrame(realDataChain)
    print("Numpy conversion")
    npy2 = rdf2.AsNumpy(columns=featureNames)
    realData = np.column_stack(list(npy2.values()))
    print("\nShape : ", signal.shape, realData.shape)
    return signal, realData


    
def getTypes():
    l=[
np.float32,   np.float32,   np.float32,   np.float32,
np.uint8,     np.int32,     np.float32,   np.float32,
np.float32,   np.float32,   np.float32,   np.float32,
np.float32,   np.uint8,     np.int32,     np.float32,
np.float32,   np.float32,   np.float32,   np.float32,
np.float32,   np.float32,   np.float32,   np.float32,
np.float32,   np.float32,   np.float32,   np.float32,
np.uint8,     np.float32,   np.float32,   np.float32,
np.float32,   np.float32,   np.float32,   np.float32,
np.float32,   np.bool_,     np.uint8,     np.uint8,
np.float32,
    ]
    return l
def getFeaturesBScoreBased(number=False, unit=False):
    featureNames = [
        'Jet1Pt',               'Jet1Eta',                  'Jet1Phi',                  'Jet1Mass',
        'Jet1_nMuons',          'Jet1_nElectrons',          'Jet1_btagDeepFlavB',       'Jet1_Area', 
        'Jet1_QGl',             'Jet2Pt',                  'Jet2Eta',                  'Jet2Phi',
        'Jet2Mass',             'Jet2_nMuons',              'Jet2_nElectrons',          'Jet2_btagDeepFlavB',
        'Jet2_Area',            'Jet2_QGl',                 'DijetPt',                 'DijetEta',
        'DijetPhi',             'DijetMass',                'dR_Dijet',                'dEta_Dijet',#r'$\Delta\eta$ Dijet',
        #r'$\Delta\varphi$ Dijet','d_Eta+Pi-dPhi',           'twist_Dijet',             'ht',
        'dPhi_Dijet','d_Eta+Pi-dPhi',           'twist_Dijet',             #'njet',
        'ht', 
        'Muon Pt',                 'Muon Eta',                 'Muon_dR_jet',
        'Muon_pt/Jet_pt',       'Muon_dxy_sig',            'Muon_dz_sig',             'Muon_ip3d',
        'Muon_sip3d',           'Muon_tightId',
        'Muon_pfIsoId',          'Muon_tkIsoId',       'SF',
        ]
    featureNames = [
        'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
        'jet1_nMuons', 'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_area', 'jet1_qgl',
        'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
        'jet2_nMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB', 'jet2_area', 'jet2_qgl',
        'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass',
        'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 'dijet_angVariable',
        'dijet_twist', 'nJets', 'ht',
        'muon_pt', 'muon_eta',
        'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d',
        'muon_tightId', 'muon_pfIsoId', 'muon_tkIsoId', 'sf']
    if number:
        for i in range(len(featureNames)):
            featureNames[i]=str(i)+"_"+featureNames[i]
    if unit:
        for i in range(len(featureNames)):
            if (featureNames[i][-2:]=="Pt") | (featureNames[i][-4:]=="Mass") | (featureNames[i]=="ht"):
                featureNames[i] = featureNames[i]+" [GeV]"
    
            
    return featureNames.copy()


def getBins(dictFormat=False):
    '''nBin, x1, x2'''
    '''bins = np.linspace(xlims[i*nCol+j,1], xlims[i*nCol+j,2], int(xlims[i*nCol+j,0])+1)'''
    bins=np.array([
        [50,0,        200,],   [50,-4,        4,],    [50,-np.pi, np.pi,],   [50,0,        30,],
        [10,-0.5,     9.5],    [6,-0.5,        5.5],  [50,0,         1],     [30,0.3,      0.7],
        [50,-0.5,      1],     [50,0,        120,],   [50,-4,        4,],    [50,-np.pi,   np.pi,],
        [50,0,         30,],   [10,-0.5,      9.5],   [5 ,-0.5,       4.5],     [50,0,        1],
        [30,0.3,       0.7],   [50,-0.5,      1],     [50,0,        300],    [50,-5,       5],
        [50,-np.pi,    np.pi], [50,0,        450],    [50,0,         6],     [50,0,        6],
        [50,0,         np.pi], [50,0,        10],     [50,0,     np.pi/2],   [16, -0.5, 31.5], [50,0,     800],
        [50,0,        40],     [50,-2.5,     2.5],         [50,-35,      35],     [50,-50,      50],     [50,-0,       0.4],
        [50,0,         100],   [2 ,-0.5,    1.5],      [3 ,-0.5,     2.5],     [3 ,-0.5,    2.5],
        #[2 ,-0.5,    1.5],     [2 ,-0.5,   1.5],      [2 ,-0.5,    1.5],     [3 ,-0.5,     2.5],
        [50 ,0,        1],

        
        ])
    
    if dictFormat:
        binsDict = {
            'jet1_pt': [50.0, 0.0, 200.0], 'jet1_eta': [50.0, -3, 3], 'jet1_phi': [50.0, -np.pi, np.pi],
            'jet1_mass': [50.0, 0.0, 30.0], 'jet1_nMuons': [10.0, -0.5, 9.5],'jet1_nTightMuons': [4.0, -0.5, 3.5], 'jet1_nElectrons': [6.0, -0.5, 5.5],
            'jet1_btagDeepFlavB': [50.0, 0.0, 1.0], 'jet1_area': [30.0, 0.3, 0.7], 'jet1_qgl': [50.0, -0.5, 1.0],
            'jet2_pt': [50.0, 0, 120.0], 'jet2_eta': [50.0, -3, 3], 'jet2_phi': [50.0, -np.pi, np.pi],
            'jet2_mass': [50.0, 0, 20], 'jet2_nMuons': [10.0, -0.5, 9.5], 'jet2_nTightMuons': [5.0, -0.5, 2.5],'jet2_nElectrons': [5.0, -0.5, 4.5],
            'jet2_btagDeepFlavB': [50.0, 0.0, 1.0], 'jet2_area': [30.0, 0.3, 0.7], 'jet2_qgl': [50.0, -0.5, 1.0],
            'jet3_pt':[30,0,100], 'jet3_eta': [50.0, -3, 3], 'jet3_phi':[30,-np.pi,np.pi],'jet3_mass':[30,0,30],'jet3_nTightMuons':[4,-.5,2.5],
            'jet3_btagDeepFlavB':[10,0,1],'jet3_btagDeepFlavC':[10,0,1],'jet3_qgl':[10,0,1],
            'dijet_pt': [50.0, 0.0, 300.0], 'dijet_eta': [50.0, -5.0, 5.0], 'dijet_phi': [50.0, -np.pi, np.pi], 
            'dijet_mass': [50.0, 0.0, 450.0], 'dijet_dR': [50.0, 0.0, 6.0], 'dijet_dEta': [50.0, 0.0, 6.0], 'dijet_dPhi': [50.0, 0.0, np.pi],
            'dijet_cs': [20, -1, 1],
            'dijet_angVariable': [50.0, 0.0, 10.0], 'dijet_twist': [50.0, 0.0, np.pi/2],'nJets': [16.0, -0.5, 31.5],
            'nJets_20GeV': [12, -0.5, 11.5], 'ht': [50.0, 0.0, 800.0], 'muon_pt': [50.0, 0.0, 40.0], 'muon_eta': [50.0, -2.5, 2.5],
            'muon_dxySig': [50.0, -35.0, 35.0], 'muon_dzSig': [50.0, -50.0, 50.0], 'muon_IP3d': [50.0, 0.0, 0.4], 'muon_sIP3d': [50.0, 0.0, 100.0],
            'muon_tightId': [2.0, -0.5, 1.5], 'muon_pfRelIso03_all': [10.0, 0, 4], 
            'muon_pfRelIso04_all':[10,0,4],
            'muon_tkIsoId': [3.0, -0.5, 2.5], 
            'muon2_pt': [50.0, 0.0, 40.0], 'muon2_eta': [50.0, -2.5, 2.5],
            'muon2_dxySig': [50.0, -35.0, 35.0], 'muon2_dzSig': [50.0, -50.0, 50.0], 'muon2_IP3d': [50.0, 0.0, 0.4], 'muon2_sIP3d': [50.0, 0.0, 100.0],
            'muon2_tightId': [2.0, -0.5, 1.5], 'muon2_pfRelIso03_all': [10.0, 0, 4], 
            'muon2_pfRelIso04_all':[10,0,4],
            'muon2_tkIsoId': [3.0, -0.5, 2.5],
            'dimuon_mass': [21, 0, 30],
            #'Muon_vx': [20, -0.50, .50],
            #'Muon_vy': [20, -0.50, .50],
            'nSV':[8,-.5, 7.5],
            'muon2_charge':[10, -1, 2],
            'Muon_fired_HLT_Mu12_IP6':[2,-.5,1.5],
            'Muon_fired_HLT_Mu8_IP3':[2,-.5,1.5],
            'Muon_fired_HLT_Mu8_IP5':[2,-.5,1.5],
            'Muon_fired_HLT_Mu8_IP6':[2,-.5,1.5],
            'Muon_fired_HLT_Mu9_IP4':[2,-.5,1.5],
            'Muon_fired_HLT_Mu9_IP5':[2,-.5,1.5],
            'Muon_fired_HLT_Mu9_IP6':[2,-.5,1.5],
            'Muon_fired_HLT_Mu7_IP4':[2,-.5,1.5],
            'PV_npvs':[50,0,49],
            'Pileup_nTrueInt': [20, -0.5, 40.5],
            'sf': [50.0, 0.0, 1.0],
            'PU_SF':[21, 0, 2]}
        return pd.DataFrame(binsDict)
    else:
        return bins.copy()


def scatter2Features(sig, bkg, labels,bins, outFile, figsize=0):
    plt.rcdefaults()
    print("Start scatter plot")
    ''' function that makes a scatter plot with all the variables given as argument for signal and background'''
    if figsize==0:
        figsize=sig.shape[1]*3
    numFeatures = sig.shape[1]
    fig, ax = plt.subplots(numFeatures, numFeatures, figsize=(figsize, figsize), constrained_layout=True)
    for i in range(numFeatures):
        
        for j in range(i+1, numFeatures):
            ax[i, j].scatter(bkg[:,j], bkg[:,i], alpha=0.5, s=2, color='red')
            ax[i, j].scatter(sig[:,j], sig[:,i], alpha=0.5, s=2, color='blue')
            ax[i, j].set_xlim(bins[j][1:])
            ax[i, j].set_ylim(bins[i][1:])

    for i in range(numFeatures):
            c, b_ = ax[i, i].hist(bkg[bkg[:,i]>-998,i], bins=int(bins[i,0]), range=bins[i,1:], label='Background', histtype=u'step' , color='red', density=True)[:2]
            ax[i, i].hist(sig[sig[:,i]>-998,i], bins=b_, label='Signal', histtype=u'step',  color='blue', density=True)[:2]
            ax[i, i].set_ylabel(labels[i], fontsize=20)
            ax[i, i].set_xlim(bins[i][1:])
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
    def load_data_generator(fileNames):
        for fileName in fileNames:
            sys.stdout.write('\r')
            sys.stdout.write("   %d/%d   "%(fileNames.index(fileName)+1, len(fileNames)))
            sys.stdout.flush()
            yield np.array(np.load(fileName, mmap_mode='r')[:, :], dtype=np.float32)
#    signal = np.load(signalFileNames[0])[:,:]
#    for signalFileName in signalFileNames[1:]:
#        sys.stdout.write('\r')
#        sys.stdout.write("   %d/%d   "%(signalFileNames.index(signalFileName)+1, len(signalFileNames)))
#        sys.stdout.flush()
#
#        currentSignal = np.load(signalFileName)[:,:]
#        signal = np.concatenate((signal, currentSignal))
    signal = np.concatenate(list(load_data_generator(signalFileNames)), axis=0)
    print("Signal shape: ", signal.shape)


    # In Python, a generator is a special type of iterator that allows you to iterate over a potentially large 
    # sequence of data without loading the entire sequence into memory at once. It generates values on-the-fly 
    # as you iterate through it, making it memory-efficient.


    bscore4 = np.concatenate(list(load_data_generator(realDataFileNames)), axis=0)
    #bscore4=bscore4[(bscore4[:,0]>20)&(bscore4[:,9]>20)]
    #bscore4 = np.load(realDataFileNames[0])[:,:]
    #for bscore4FileName in realDataFileNames[1:]:
    #    sys.stdout.write('\r')
    #    sys.stdout.write("   %d/%d   "%(realDataFileNames.index(bscore4FileName)+1, len(realDataFileNames)))
    #    sys.stdout.flush()
    #    try:
    #        currentBscore4 = np.load(bscore4FileName)[:,:]
    #    except:
    #        print(bscore4FileName)
    #        
    #    if currentBscore4[0,21]==0:
    #        print(bscore4FileName)
    #    try:
    #        bscore4 = np.concatenate((bscore4, currentBscore4))
    #    except:
    #        print(bscore4FileName)
    print("bscore4 shape: ", bscore4.shape)

    return signal, bscore4
def loadDataOnlyFeatures(signalPath, realDataPath, nSignalFiles=-1, nRealDataFiles=-1, features=[21,-1]):
    print("Loading Data...")
    signalFileNames = glob.glob(signalPath+"/*bScoreBased4_*.npy")
    realDataFileNames = glob.glob(realDataPath+"/*bScoreBased4_*.npy")
    signalFileNames = signalFileNames[:nSignalFiles] if nSignalFiles!=-1 else signalFileNames
    realDataFileNames = realDataFileNames[:nRealDataFiles] if nRealDataFiles!=-1 else realDataFileNames

    def load_data_generator(fileNames):
        for fileName in fileNames:
            sys.stdout.write('\r')
            sys.stdout.write("   %d/%d   "%(fileNames.index(fileName)+1, len(fileNames)))
            sys.stdout.flush()
            yield np.load(fileName)[:, features]

    print("%d files for MC ggHbb" %len(signalFileNames))
    print("%d files for realDataFileNames" %len(realDataFileNames))
    signal = np.concatenate(list(load_data_generator(signalFileNames)), axis=0)
    #signal = np.load(signalFileNames[0])[:,features]
    #for signalFileName in signalFileNames[1:]:
    #    sys.stdout.write('\r')
    #    sys.stdout.write("   %d/%d   "%(signalFileNames.index(signalFileName)+1, len(signalFileNames)))
    #    sys.stdout.flush()
#
    #    currentSignal = np.load(signalFileName)[:,features]
    #    signal = np.concatenate((signal, currentSignal))
    print("Signal shape: ", signal.shape)
    bscore4 = np.concatenate(list(load_data_generator(realDataFileNames)), axis=0)
    #bscore4 = np.load(realDataFileNames[0])[:,features]
    #for bscore4FileName in realDataFileNames[1:]:
    #    sys.stdout.write('\r')
    #    sys.stdout.write("   %d/%d   "%(realDataFileNames.index(bscore4FileName)+1, len(realDataFileNames)))
    #    sys.stdout.flush()
    #    try:
    #        currentBscore4 = np.load(bscore4FileName)[:,features]
    #    except:
    #        print(bscore4FileName)
    #    m = currentBscore4[:,0]<1
    #    if np.sum(m)>100:
    #        print(bscore4FileName)
    #    try:
    #        bscore4 = np.concatenate((bscore4, currentBscore4))
    #    except:
    #        print(bscore4FileName)
    print("bscore4 shape: ", bscore4.shape)

    return signal, bscore4

def loadDataOnlyMass(signalPath, realDataPath, nSignalFiles, nRealDataFiles):
    print("Loading Data...")
    signalFileNames = glob.glob(signalPath+"/*bScoreBased4_*.npy")
    realDataFileNames = glob.glob(realDataPath+"/*bScoreBased4_*.npy")
    signalFileNames = signalFileNames[:nSignalFiles] if nSignalFiles!=-1 else signalFileNames
    realDataFileNames = realDataFileNames[:nRealDataFiles] if nRealDataFiles!=-1 else realDataFileNames

    print("%d files for MC ggHbb" %len(signalFileNames))
    print("%d files for realDataFileNames" %len(realDataFileNames))
    signal = np.load(signalFileNames[0])[:,21]
    for signalFileName in signalFileNames[1:]:
        sys.stdout.write('\r')
        sys.stdout.write("   %d/%d   "%(signalFileNames.index(signalFileName)+1, len(signalFileNames)))
        sys.stdout.flush()

        currentSignal = np.load(signalFileName)[:,21]
        signal = np.concatenate((signal, currentSignal))
    print("Signal shape: ", signal.shape)

    bscore4 = np.load(realDataFileNames[0])[:,21]
    for bscore4FileName in realDataFileNames[1:]:
        sys.stdout.write('\r')
        sys.stdout.write("   %d/%d   "%(realDataFileNames.index(bscore4FileName)+1, len(realDataFileNames)))
        sys.stdout.flush()
        try:
            currentBscore4 = np.load(bscore4FileName)[:,21]
        except:
            print(bscore4FileName)
        if ((currentBscore4[0]==0) & (currentBscore4[1]==0)):
            print(bscore4FileName)
            continue
        try:
            bscore4 = np.concatenate((bscore4, currentBscore4))
        except:
            print(bscore4FileName)
    print("bscore4 shape: ", bscore4.shape)

    return signal, bscore4


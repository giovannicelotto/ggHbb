import glob, re, sys
sys.path.append('/t3home/gcelotto/ggHbb/NN')
from applyMultiClass_Hpeak import getPredictions, splitPtFunc
from functions import loadMultiParquet, cut, getXSectionBR
from helpersForNN import preprocessMultiClass, scale, unscale
import numpy as np
def loadData(pdgID=25, nReal = -2):
    pTClass, nMC = 0,-1
    if pdgID==25:
        paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB",
            ]
        isMCList = [1]
    elif pdgID==23:
        paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
            ]
        isMCList = [36, 20, 21, 22, 23]
    elif pdgID==0:
        paths = ['/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others']
        isMCList = [0]
    pathToPredictions = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/NNpredictions"
    # check for which fileNumbers the predictions is available
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
    dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt','jet1_mass', 'jet2_mass',
                                                                                                      'jet1_eta', 'jet2_eta', 'jet1_qgl', 'jet2_qgl', 'Pileup_nTrueInt', 'jet3_pt', 'jet3_mass',
                                                                                                      'jet2_btagDeepFlavB'],
                                                                                                      returnNumEventsTotal=True, selectFileNumberList=fileNumberList, returnFileNumberList=True)

    import json
    def load_mapping_dict(file_path):
        with open(file_path, 'r') as file:
            mapping_dict = json.load(file)
            # Convert keys back to integers if needed (they should already be integers)
            mapping_dict = {int(k): v for k, v in mapping_dict.items()}

        return mapping_dict
    PU_map = load_mapping_dict('/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_PVtoPUSF.json')
    for df in dfs:
        #print(df['Pileup_nTrueInt'])
        df['PU_SF'] = df['Pileup_nTrueInt'].apply(int).map(PU_map)
        df.loc[df['Pileup_nTrueInt'] > 98, 'PU_SF'] = 0


    pTmin, pTmax, suffix = [[0,-1,'inclusive'], [0, 30, 'lowPt'], [30, 100, 'mediumPt'], [100, -1, 'highPt']][pTClass]    
    for df in dfs:
        print()
    print(pTmin, pTmax, suffix)
    dfs = preprocessMultiClass(dfs, None, pTmin, pTmax, suffix)   # get the dfs with the cut in the pt class

    minPt, maxPt = None, None #180, -1
    if (minPt is not None) | (maxPt is not None):
        dfs, masks = splitPtFunc(dfs, minPt, maxPt)
        splitPt = True
    else:
        masks=None
        splitPt=False




    if pdgID==25:

        YPred_H = getPredictions(fileNumberList, pathToPredictions, splitPt=splitPt, masks=masks, isMC=isMCList, pTClass=pTClass)
        YPred_H = np.array(YPred_H)[0]
        return dfs, YPred_H
    elif pdgID==23:
        YPred_Z100, YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800 = getPredictions(fileNumberList, pathToPredictions, splitPt=splitPt, masks=masks, isMC=isMCList, pTClass=pTClass)

        return dfs, YPred_Z100, YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800, numEventsList
    elif pdgID==0:
        YPred_QCD = getPredictions(fileNumberList, pathToPredictions, splitPt=splitPt, masks=masks, isMC=isMCList, pTClass=pTClass)
        YPred_QCD = np.array(YPred_QCD)[0]
        return dfs, YPred_QCD, numEventsList
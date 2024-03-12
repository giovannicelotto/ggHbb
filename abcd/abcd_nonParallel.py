import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
def getProcessesDataFrame():
    nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    processes = {
        #'BParkingDataRun20181A':            [nanoPathCommon + "/Data20181A_2023Nov30",                                                                                flatPathCommon + "/Data1A"    ,                   -1],
        #'GluGluHToBB':                      [nanoPathCommon + "/GluGluHToBB_20UL18",                                                                                  flatPathCommon + "/GluGluHToBB"    ,      30.52],
        #'WW':                               [nanoPathCommon + "/diboson2024Feb14/WW_TuneCP5_13TeV-pythia8",                                                           flatPathCommon + "/diboson/WW",                   75.8],
        #'WZ':                               [nanoPathCommon + "/diboson2024Feb14/WZ_TuneCP5_13TeV-pythia8",                                                           flatPathCommon + "/diboson/WZ",                   27.6],
        #'ZZ':                               [nanoPathCommon + "/diboson2024Feb14/ZZ_TuneCP5_13TeV-pythia8",                                                           flatPathCommon + "/diboson/ZZ",                   12.14	],
        #'ST_s-channel-hadronic':            [nanoPathCommon + "/singleTop2024Feb14/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",                    flatPathCommon + "/singleTop/s-channel_hadronic", 11.24],
        #'ST_s-channel-leptononic':          [nanoPathCommon + "/singleTop2024Feb14/ST_s-channel_4f_leptonDecays_TuneCP5CR1_13TeV-amcatnlo-pythia8",                   flatPathCommon + "/singleTop/s-channel_leptonic", 3.74],
        #'ST_t-channel-antitop':             [nanoPathCommon + "/singleTop2024Feb14/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5CR1_13TeV-powheg-madspin-pythia8",  flatPathCommon + "/singleTop/t-channel_antitop" , 69.09],
        #'ST_t-channel-top':                 [nanoPathCommon + "/singleTop2024Feb14/ST_t-channel_top_4f_InclusiveDecays_TuneCP5CR1_13TeV-powheg-madspin-pythia8",      flatPathCommon + "/singleTop/t-channel_top"     , 115.3],
        #'ST_tW-antitop':                    [nanoPathCommon + "/singleTop2024Feb14/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                    flatPathCommon + "/singleTop/tW-channel_antitop", 34.97],
        #'ST_tW-top':                        [nanoPathCommon + "/singleTop2024Feb14/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                        flatPathCommon + "/singleTop/tW-channel_top"    , 34.91	],
        #'TTTo2L2Nu':                        [nanoPathCommon + "/ttbar2024Feb14/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",                                               flatPathCommon + "/ttbar/ttbar2L2Nu"            , 831*0.09],
        #'TTToHadronic':                     [nanoPathCommon + "/ttbar2024Feb14/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",                                            flatPathCommon + "/ttbar/ttbarHadronic"         , 831*0.46],
        #'TTToSemiLeptonic':                 [nanoPathCommon + "/ttbar2024Feb14/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",                                        flatPathCommon + "/ttbar/ttbarSemiLeptonic"     , 831*0.45],
        #'WJetsToLNu':                       [nanoPathCommon + "/WJetsToLNu2024Feb20/*",                                                                               flatPathCommon + "/WJets/WJetsToLNu"            , 62070.0],
        #'WJetsToQQ_200to400'      :         [nanoPathCommon + "/WJetsToQQ2024Feb20/WJetsToQQ_HT-200to400*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-200to400" , 2549.0],
        #'WJetsToQQ_400to600'      :         [nanoPathCommon + "/WJetsToQQ2024Feb20/WJetsToQQ_HT-400to600*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-400to600" , 276.5],
        #'WJetsToQQ_600to800'      :         [nanoPathCommon + "/WJetsToQQ2024Feb20/WJetsToQQ_HT-600to800*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-600to800" , 59.25],
        #'WJetsToQQ_800toInf'      :         [nanoPathCommon + "/WJetsToQQ2024Feb20/WJetsToQQ_HT-800toInf*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-800toInf" , 28.75],
        #'ZJetsToQQ_200to400'      :         [nanoPathCommon + "/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-200to400*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-200to400" , 1012.0],
        #'ZJetsToQQ_400to600'      :         [nanoPathCommon + "/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-400to600*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-400to600" , 114.2],
        #'ZJetsToQQ_600to800'      :         [nanoPathCommon + "/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-600to800*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-600to800" , 25.34],
        #'ZJetsToQQ_800toInf'      :         [nanoPathCommon + "/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-800toInf*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-800toInf" , 12.99],
        #'QCD_MuEnriched_Pt-1000':           [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-1000*",                                                                flatPathCommon + "/QCD_Pt1000ToInf"             , 1.085],
        #'QCD_MuEnriched_Pt-800To1000':      [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-800To1000*",                                                           flatPathCommon + "/QCD_Pt800To1000"             , 3.318],
        #'QCD_MuEnriched_Pt-600To800':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-600To800*",                                                            flatPathCommon + "/QCD_Pt600To800"              , 18.12	],
        #'QCD_MuEnriched_Pt-470To600':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-470To600*",                                                            flatPathCommon + "/QCD_Pt470To600"              , 58.9],
        #'QCD_MuEnriched_Pt-300To470':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-300To470*",                                                            flatPathCommon + "/QCD_Pt300To470"              , 622.6],
        #'QCD_MuEnriched_Pt-170To300':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-170To300*",                                                            flatPathCommon + "/QCD_Pt170To300"              , 7000.0],
        #'QCD_MuEnriched_Pt-120To170':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-120To170*",                                                            flatPathCommon + "/QCD_Pt120To170"              , 21280.0],
        #'QCD_MuEnriched_Pt-80To120':        [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-80To120*",                                                             flatPathCommon + "/QCD_Pt80To120"               , 87740.0],
        #'QCD_MuEnriched_Pt-50To80':         [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-50To80*",                                                              flatPathCommon + "/QCD_Pt50To80"                , 381700.0],
        'QCD_MuEnriched_Pt-30To50':         [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-30To50*",                                                              flatPathCommon + "/QCD_Pt30To50"                , 1367000.0],
        'QCD_MuEnriched_Pt-20To30':         [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-20To30*",                                                              flatPathCommon + "/QCD_Pt20To30"                , 2527000.0],
        'QCD_MuEnriched_Pt-15To20':         [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-15To20*",                                                              flatPathCommon + "/QCD_Pt15To20"                , 2800000.0	],

    }

    df = pd.DataFrame(processes).T
    df.columns = ['nanoPath', 'flatPath', 'xsection']
    return df
def main(nFilesData, nFilesMC):
    df = getProcessesDataFrame()
    df.to_csv("/t3home/gcelotto/ggHbb/abcd/output/processes.csv")
    #currentLumi = nFilesData * 0.874 / 1017
    #np.save('/t3home/gcelotto/ggHbb/abcd/output/currentLumi.npy', currentLumi)
    
    #variable x1 and x2
    x1_threshold = 0.5
    x2_threshold = 1
    x1 = 'jet1_btagDeepFlavB'
    x2 = 'dijet_twist'
    # main variable and binning
    xx = 'dijet_mass'
    bins = np.linspace(0, 300, 20)

    miniDf = pd.read_csv("/t3home/gcelotto/ggHbb/abcd/output/miniDf.csv")
    regions = {
        'A':np.zeros(len(bins)-1),
        'B':np.zeros(len(bins)-1),
        'D':np.zeros(len(bins)-1),
        'C':np.zeros(len(bins)-1),
    }
    for (process, flatPath, xsection) in zip(df.index, df.flatPath, df.xsection):
        doneFiles = 0
        numEventsPassedFiles = 0
        print("Starting ", process)
        flatFileNames = glob.glob(flatPath+"/**/*.parquet", recursive=True)
        
        print("%d files"%len(flatFileNames))
        for flatFileName in flatFileNames:
            if (doneFiles==nFilesData) & (process=="BParkingDataRun20181A"):
                break
            elif (doneFiles==nFilesMC) & (process!="BParkingDataRun20181A"):
                break
            print("Opening ", flatFileName)
            fileNumber = re.search(r'\D(\d{1,4})\.\w+$', flatFileName).group(1)
            if process!='BParkingDataRun20181A':
                print(fileNumber, process)
                mini = miniDf[(miniDf.fileNumber==int(fileNumber)) & (miniDf.process==process)].numEventsTotal.iloc[0]
                
                numEventsPassedFiles = numEventsPassedFiles + mini
            df = pd.read_parquet(flatFileName, columns=['sf', 'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'dijet_twist', 'dijet_mass'])
            
            regions['A'] = regions['A'] + np.histogram(df[(df[x1]<x1_threshold)  & (df[x2]>=  x2_threshold)][xx], bins=bins, weights=df.sf[(df[x1]< x1_threshold)  & (df[x2]>=  x2_threshold)])[0]
            regions['B'] = regions['B'] + np.histogram(df[(df[x1]>=x1_threshold) & (df[x2]>=  x2_threshold)][xx], bins=bins, weights=df.sf[(df[x1]>=x1_threshold)  & (df[x2]>=  x2_threshold)])[0]
            regions['D'] = regions['D'] + np.histogram(df[(df[x1]>=x1_threshold) & (df[x2]<   x2_threshold)][xx], bins=bins, weights=df.sf[(df[x1]>=x1_threshold)  & (df[x2] <  x2_threshold)])[0]
            regions['C'] = regions['C'] + np.histogram(df[(df[x1]<x1_threshold)  & (df[x2]<   x2_threshold)][xx], bins=bins, weights=df.sf[(df[x1]< x1_threshold)  & (df[x2] <  x2_threshold)])[0]
            doneFiles = doneFiles+1
        if process!='BParkingDataRun20181A':
            regions['A'] = regions['A']*xsection/numEventsPassedFiles
            regions['B'] = regions['B']*xsection/numEventsPassedFiles
            regions['D'] = regions['D']*xsection/numEventsPassedFiles
            regions['C'] = regions['C']*xsection/numEventsPassedFiles
        else:
            pass
    print('A', regions["A"])
    print('B', regions["B"])
    print('D', regions['D'])
    print('C', regions['C'])

    print("\nBKG estimation")
    print("B", regions['A']*regions['D']/regions['C'], )

    fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(15, 10))
    x=(bins[1:]+bins[:-1])/2
    ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"], histtype=u'step', label='Region A')
    ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"], histtype=u'step', label='Region B')
    ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'], histtype=u'step', label='Region D')
    ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'], histtype=u'step', label='Region C')
    
    ax[0,1].hist(bins[:-1], bins=bins, weights=regions['A']*regions['D']/regions['C'], histtype=u'step', label=r'$A\times D / C$ ')
    ax[0,2].hist(bins[:-1], bins=bins, weights=regions["B"]*regions['C']/(regions['A']*regions['D']), histtype=u'step', label=r'B$\times$C/A$\times$D')


    ax[0,0].set_title("%s < %.1f, %s >= %.1f"%(x1, x1_threshold, x2, x2_threshold), fontsize=14)
    ax[0,1].set_title("%s >= %.1f, %s >= %.1f"%(x1, x1_threshold, x2, x2_threshold), fontsize=14)
    ax[1,0].set_title("%s < %.1f, %s < %.1f"%(x1, x1_threshold, x2, x2_threshold), fontsize=14)
    ax[1,1].set_title("%s >= %.1f, %s < %.1f"%(x1, x1_threshold, x2, x2_threshold), fontsize=14)
    for idx, axx in enumerate(ax.ravel()):
        axx.set_xlim(bins[0], bins[-1])
        axx.set_xlabel("Dijet Mass [GeV]")
        axx.legend(fontsize=18, loc='upper right')
    fig.savefig("", bbox_inches='tight')
    print("Saving in ", "/t3home/gcelotto/ggHbb/abcd/output/abcd.png")
    return 

if __name__ == "__main__":
    nFilesData, nFilesMC = sys.argv[1:]
    nFilesData, nFilesMC = int(nFilesData), int(nFilesMC)
    main(nFilesData, nFilesMC)
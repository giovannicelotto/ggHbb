# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functions import *
import mplhep as hep
from scipy.stats import shapiro, kstest, norm, chi2
hep.style.use("CMS")
def plotDataMC(dfMC, dfData, feature, bins, showChi2=False, outName=None):
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])


    x = (bins[1:] + bins[:-1])/2
    ax[0].set_xlim(bins[0], bins[-1])
    cData=np.histogram(dfData[feature], bins=bins)[0]
    ax[0].errorbar(x, cData, np.sqrt(cData), marker='o', color='black', linestyle='none', label='Data')


    cumulativeMC = np.zeros(len(x))
    cMC_err = np.zeros(len(x))
    for processPlot in np.unique(dfMC.process):
        #print(processPlot)
        df_ = dfMC[dfMC.process==processPlot]
        cMC=ax[0].hist(df_[feature], bins=bins, weights=df_.weight, bottom=cumulativeMC, label=processPlot)[0]
        cMC_err+=np.histogram(df_[feature], bins=bins, weights=df_.weight**2)[0]
        cumulativeMC=cumulativeMC+cMC
    cMC_err=np.sqrt(cMC_err)
    ax[0].legend()

    if showChi2:
        chi2_stat = np.sum((cData[cData>0] - cumulativeMC[cData>0])**2/(cData[cData>0]+cMC_err[cData>0]**2))
        ndof = np.sum(cData>0)
        chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
        ax[0].text(x=0.95, y=0.6, s="Chi2/ndof = %.2f/%d (p = %.2f)"%(chi2_stat, ndof, chi2_pvalue) , transform=ax[0].transAxes, ha='right')
    ax[1].errorbar(x, cData/(cumulativeMC+1e-12), yerr=np.sqrt(cData)/(cumulativeMC+1e-12), linestyle='none', marker='o', color='black')
    err_band = cMC_err / (cumulativeMC+1e-12)
    err_band_edges = np.repeat(err_band, 2)  # duplicate each bin's error
    x_edges = np.repeat(bins, 2)[1:-1]

    ax[1].fill_between(
        x_edges,
        1 - err_band_edges,
        1 + err_band_edges,
        step='post',
        color='gray',
        alpha=0.5,
        hatch='///',
        label='MC stat. unc.'
    )
    ax[1].set_ylim(0., 2.)
    ax[1].set_xlim(ax[1].get_xlim())
    ax[1].set_xlabel(feature)
    hep.cms.label(lumi=np.round(lumi, 2), ax=ax[0])
    ax[1].hlines(y=1,xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')
    if outName is None:
        fig.savefig("/t3home/gcelotto/ggHbb/tt_CR/plots/%s.png"%feature, bbox_inches='tight')
    else:
        fig.savefig(outName, bbox_inches='tight')
    plt.close('all')
    return cData, cumulativeMC
# %%
isMCList = [11,12,13, 5, 6, 7, 8, 9, 10]
nMCs = 100
dfsMC, sumw = loadMultiParquet_v2(paths=isMCList, nMCs=nMCs,columns=None, returnNumEventsTotal=True, filters=getCommonFilters(btagWP="T"))
# %%
dfProcesses = getDfProcesses_v2()[0]
# %%
dfsData, lumi = loadMultiParquet_Data_new(dataTaking=[
    # Data *A
    0, 1, 2 ,3, 4, 5,
    # Data *B
    6, 7, 8, 9, 10,11,
    ## Data *C
    12, 13, 14, 15, 16,
    ## Data *D
    #17,
    18,19,20, 21
    ], nReals=100, columns=None, filters=getCommonFilters(btagWP="T", cutDijet=True))
# %%

dfsData_ = dfsData.copy()
dfsMC_ = dfsMC.copy()
# %%
dfsData = cut(dfsData, 'dijet_pt', 100, None)
dfsMC = cut(dfsMC, 'dijet_pt', 100, None)
# %%
dfsData = cut(dfsData, 'Muon_tt_pt', 0, None)
dfsMC = cut(dfsMC, 'Muon_tt_pt', 0, None)
# %%
dfsData = cut(dfsData, 'muon_pt', 9, None)
dfsMC = cut(dfsMC, 'muon_pt', 9, None)
# %%
dfsData = cut(dfsData, 'dilepton_tt_mass', 20, None)
dfsMC = cut(dfsMC, 'dilepton_tt_mass', 20, None)

dfsData = cut(dfsData, 'jet1_pt_uncor', 20, None)
dfsData = cut(dfsData, 'jet1_pt_uncor', 20, None)
dfsMC = cut(dfsMC, 'jet1_pt_uncor', 20, None)
dfsMC = cut(dfsMC, 'jet1_pt_uncor', 20, None)

# %%
for idx, df in enumerate(dfsMC):
    dfsMC[idx]['weight'] = dfsMC[idx]['genWeight'] * dfsMC[idx]['btag_central'] * dfsMC[idx]['sf'] * dfsMC[idx]['PU_SF'] * dfsMC[idx]["jet_pileupId_SF_nom"] * dfsMC[idx]['Muon_tt_RECO_SF'] * dfsMC[idx]['Muon_tt_ID_SF'] * dfsMC[idx]['Muon_tt_ISO_SF'] * dfsMC[idx]["Electron_tt_SF"] * lumi * 1000 * dfProcesses.iloc[isMCList[idx]].xsection / sumw[idx] 
    if dfProcesses.iloc[isMCList[idx]].process[:2]=='ST':
        dfsMC[idx]['process'] = 'ST'
    elif dfProcesses.iloc[isMCList[idx]].process[:5]=='TTTo2':
        dfsMC[idx]['process'] = 'tt (ll)'
    elif (dfProcesses.iloc[isMCList[idx]].process[:8]=='TTToHadr') | (dfProcesses.iloc[isMCList[idx]].process[:8]=='TTToSemi'):
        dfsMC[idx]['process'] = 'tt (others)'
    else:
        dfsMC[idx]['process'] = 'Others'
    if len(dfsMC[idx])>0:
        print("Process : ", dfProcesses.iloc[isMCList[idx]].process, " in ", np.unique(dfsMC[idx]['process']), "\n  events : %d \n  xsection : %.3f"%(len(dfsMC[idx]), dfProcesses.iloc[isMCList[idx]].xsection) )
    else:
        print("Process : ", dfProcesses.iloc[isMCList[idx]].process, " with 0 events")

dfMC = pd.concat(dfsMC)
dfData = pd.concat(dfsData)
# %%

deltaPhi = dfMC.Muon_tt_phi  - dfMC.Electron_tt_phi
dfMC['dilepton_dPhi'] =  deltaPhi - 2*np.pi*(deltaPhi >= np.pi) + 2*np.pi*(deltaPhi< -np.pi)
deltaPhi = dfData.Muon_tt_phi  - dfData.Electron_tt_phi
dfData['dilepton_dPhi'] =  deltaPhi - 2*np.pi*(deltaPhi >= np.pi) + 2*np.pi*(deltaPhi< -np.pi)


dfMC['dilepton_dR'] = np.sqrt(dfMC.dilepton_dPhi**2 + (dfMC.Muon_tt_eta-dfMC.Electron_tt_eta)**2)
dfData['dilepton_dR'] =  np.sqrt(dfData.dilepton_dPhi**2 + (dfData.Muon_tt_eta-dfData.Electron_tt_eta)**2)

features = {
    'Electron_tt_pt':np.linspace(0, 200, 21),
    'Electron_tt_eta':np.linspace(-3, 3, 21),
    'Electron_tt_phi':np.linspace(-3.14, 3.14, 21),
    'dilepton_tt_mass':np.linspace(20, 300, 21),
    'dijet_mass':np.linspace(20, 300, 21),

    'dilepton_dR': np.linspace(0, 4, 51),

    'Muon_tt_pt':np.linspace(0, 200, 21),
    'Muon_tt_eta':np.linspace(-3, 3, 21),
    'Muon_tt_phi':np.linspace(-3.14, 3.14, 21),
    'Muon_tt_pfIsoId':np.linspace(4, 7, 21),
}

for feature, bins in zip(features.keys(), features.values()) :
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])

    x = (bins[1:] + bins[:-1])/2
    ax[0].set_xlim(bins[0], bins[-1])
    cData=np.histogram(dfData[feature], bins=bins)[0]
    ax[0].errorbar(x, cData, np.sqrt(cData), marker='o', color='black', linestyle='none', label='Data')

    bottom = np.zeros(len(x))
    cMC_err = np.zeros(len(x))
    for processPlot in np.unique(dfMC.process):
        df_ = dfMC[dfMC.process==processPlot]
        cMC=ax[0].hist(df_[feature], bins=bins, weights=df_.weight, bottom=bottom, label=processPlot)[0]
        cMC_err+=np.histogram(df_[feature], bins=bins, weights=df_.weight**2)[0]
        bottom=bottom+cMC
    cMC_err=np.sqrt(cMC_err)
    ax[0].legend()

    ax[1].errorbar(x, cData/(bottom+1e-12), yerr=np.sqrt(cData)/(bottom+1e-12), linestyle='none', marker='o', color='black')
    err_band = cMC_err / (bottom+1e-12)
    err_band_edges = np.repeat(err_band, 2)  # duplicate each bin's error
    x_edges = np.repeat(bins, 2)[1:-1]

    ax[1].fill_between(
        x_edges,
        1 - err_band_edges,
        1 + err_band_edges,
        step='post',
        color='gray',
        alpha=0.5,
        hatch='///',
        label='MC stat. unc.'
    )
    ax[1].set_ylim(0., 2)
    ax[1].set_xlim(ax[1].get_xlim())
    ax[1].set_xlabel(feature)
    hep.cms.label(lumi=np.round(lumi, 2), ax=ax[0])
    ax[1].hlines(y=1,xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')
    fig.savefig("/t3home/gcelotto/ggHbb/tt_CR/plots/%s.png"%feature, bbox_inches='tight')
    plt.close('all')
# %%
import sys
from datetime import datetime
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.getInfolderOutfolder import getInfolderOutfolder
import torch
from helpers.scaleUnscale import scale, unscale
current_date = "Aug28"  # This gives the format like 'Dec12'
boosted = 3
version = 20.01
modelName = "model.pth"
inFolder_, outFolder = getInfolderOutfolder(name = "%s_%d_%s"%(current_date, boosted, str(version).replace('.', 'p')), suffixResults='_mjjDisco', createFolder=False)
featuresForTraining = list(np.load(outFolder+"/featuresForTraining.npy"))
model = torch.load(outFolder+"/model/%s"%modelName, map_location=torch.device('cpu'), weights_only=False)
model.eval()


XDataScaled = scale(dfData,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
# %%
XMCScaled = scale(dfMC.drop(labels=['process'], axis=1),featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)

# %%
XData_tensor = torch.tensor(np.float32(XDataScaled[featuresForTraining].values)).float()
XMC_tensor = torch.tensor(np.float32(XMCScaled[featuresForTraining].values)).float()

# %%
with torch.no_grad():  # No need to track gradients for inference
    YPredData = model(XData_tensor).numpy()
    YPredMC = model(XMC_tensor).numpy()
# %%
dfMC['PNN'] = YPredMC
dfData['PNN'] = YPredData
cData, cMC = plotDataMC(dfMC=dfMC, dfData=dfData, feature='PNN', bins=np.array([0, 0.83, 1]), showChi2=True, outName=None)

epsi_MC = cMC[1]/(cMC[1]+cMC[0])
epsi_Data = cData[1]/(cData[1]+cData[0])
print("ε(MC)   \t = %.1f/%.1f = %.1f%%"%(cMC[1], cMC[1]+cMC[0], epsi_MC*100))
print("ε(Data) \t = %.1f/%.1f = %.1f%%"%(cData[1], cData[1]+cData[0], epsi_Data*100))
binom_error_epsi_MC = np.sqrt(np.sum(dfMC.weight**2)*epsi_MC * (1-epsi_MC)/np.sum(dfMC.weight)**2)
binom_error_epsi_Data = np.sqrt(epsi_Data * (1-epsi_Data)/(cData[1]+cData[0]))
print("Binomial Error for MC", binom_error_epsi_MC)
print("Binomial Error for Data", binom_error_epsi_Data)

syst = epsi_Data/epsi_MC
error_syst = syst * np.sqrt((binom_error_epsi_MC/epsi_MC)**2 + (binom_error_epsi_Data/epsi_Data)**2)
print("Syst is %.4f +- %.4f"%(syst, error_syst))

cDataFine, cMCFine = plotDataMC(dfMC=dfMC, dfData=dfData, feature='PNN', bins=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.78,0.83,1.0]), showChi2=False, outName="/t3home/gcelotto/ggHbb/tt_CR/plots/PNN_fine.png")
np.save("/t3home/gcelotto/ggHbb/tt_CR/plots/cData_PNN_fine.npy", cDataFine)
np.save("/t3home/gcelotto/ggHbb/tt_CR/plots/cMC_PNN_fine.npy", cMCFine)
#dfData = cut([dfData], 'PNN', 0.7, None)[0]
#dfMC = cut([dfMC], 'PNN', 0.7, None)[0]
#plotDataMC(dfMC=dfMC, dfData=dfData, feature='dijet_mass', bins=np.linspace(0, 500, 11), showChi2=False, outName="/t3home/gcelotto/ggHbb/tt_CR/plots/dijet_mass_nncut.png")
# %%
#trainData["PNN"] = YPredTrainData
importantFeatures = ['jet1_pt',  'jet1_mass','jet1_nConstituents',
                     'jet2_pt', 'jet2_mass','jet2_nConstituents',
                     'muon_pt',  'muon_dxySig', 'jet3_btagWP',
                     'PNN'
                     #'jet3_pt', 'jet3_eta', 'jet3_phi'
                     ]
#plotNormalizedFeatures(data=[dfMC[dfMC.PNN>0.7][importantFeatures], trainData[trainData.PNN>0.7][importantFeatures]], outFile="/t3home/gcelotto/ggHbb/tt_CR/plots/features_high.png",
#                       legendLabels=["MC", "Data"], colors=['red', 'blue'], weights=[dfMC.weight[dfMC.PNN>0.7], np.ones(np.sum(trainData.PNN>0.7))], error=False)
#
#plotNormalizedFeatures(data=[dfMC[dfMC.PNN<0.7][importantFeatures], trainData[trainData.PNN<0.7][importantFeatures]], outFile="/t3home/gcelotto/ggHbb/tt_CR/plots/features_low.png",
#                       legendLabels=["MC", "Data"], colors=['red', 'blue'], weights=[dfMC.weight[dfMC.PNN<0.7], np.ones(np.sum(trainData.PNN<0.7))], error=False)
# %%

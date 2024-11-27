# %%
import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt
hep.style.use("CMS")
from functions import loadMultiParquet, getZXsections, getXSectionBR,cut
# %%
nReal = 50
nMC = -1
columnsToRead = ['jet1_pt', 'jet2_pt', 'muon_pt', 'sf', 'PU_SF', 'jet1_id', 'jet2_id', 'muon_eta', 'Muon_fired_HLT_Mu9_IP6']
flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
paths = [
        flatPathCommon + "/Data1A/training",
        flatPathCommon + "/GluGluHToBB/training",
        flatPathCommon + "/ZJets/ZJetsToQQ_HT-100to200",
        flatPathCommon + "/ZJets/ZJetsToQQ_HT-200to400",
        flatPathCommon + "/ZJets/ZJetsToQQ_HT-400to600",
        flatPathCommon + "/ZJets/ZJetsToQQ_HT-600to800",
        flatPathCommon + "/ZJets/ZJetsToQQ_HT-800toInf"]
m=200
paths.append(flatPathCommon + "/GluGluH_M%d_ToBB"%(m))
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=columnsToRead, returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)

# %%
dfs = cut(dfs, 'Muon_fired_HLT_Mu9_IP6', 0.5, None)
#dfs = cut(dfs, 'jet1_pt', 20, None)
#dfs = cut(dfs, 'jet2_pt', 20, None)
# %%
W_Z = []
for idx, df in enumerate(dfs[2:-1]):
    w = df.sf*df.PU_SF*getZXsections()[idx]/numEventsList[idx+2]
    W_Z.append(w)
W_Z=np.concatenate(W_Z)
W_H = dfs[1].sf*dfs[1].PU_SF*getXSectionBR()/numEventsList[1]
W_Res = dfs[-1].sf*dfs[-1].PU_SF/numEventsList[-1]

dfH = dfs[1]
dfZ = pd.concat(dfs[2:-1])
dfRes = dfs[-1]
dfdata = dfs[0]
# %%


fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 60, 41)
cdata = np.histogram(np.clip(dfdata.muon_pt, bins[0], bins[-1]), bins=bins)[0]
cH = np.histogram(np.clip(dfH.muon_pt, bins[0], bins[-1]), bins=bins, weights=W_H)[0]
cZ = np.histogram(np.clip(dfZ.muon_pt, bins[0], bins[-1]), bins=bins, weights=W_Z)[0]
cRes = np.histogram(np.clip(dfRes.muon_pt, bins[0], bins[-1]), bins=bins, weights=W_Res)[0]

cdatan, cHn, cZn, cResn = cdata/np.sum(cdata), cH/np.sum(cH), cZ/np.sum(cZ), cRes/np.sum(cRes)

ax.hist(bins[:-1], bins=bins, weights=cdatan, label="Data", histtype='step')
ax.hist(bins[:-1], bins=bins, weights=cHn, label="ggH", histtype='step')
ax.hist(bins[:-1], bins=bins, weights=cZn, label="ZJets", histtype='step')
ax.hist(bins[:-1], bins=bins, weights=cResn, label="ggSpin0(200)", histtype='step')
ax.set_xlabel("Muon pt [GeV]")
ax.legend()

# mask with Muon > X GeV
# %%
threshold = 14

mdata = dfdata.muon_pt>threshold
mH = dfH.muon_pt>threshold
mZ = dfZ.muon_pt>threshold
mRes = dfRes.muon_pt>threshold
cdata_cut = np.histogram(np.clip(dfdata[mdata].muon_pt, bins[0], bins[-1]), bins=bins)[0]
cH_cut = np.histogram(np.clip(dfH[mH].muon_pt, bins[0], bins[-1]), bins=bins, weights=W_H[mH])[0]
cZ_cut = np.histogram(np.clip(dfZ[mZ].muon_pt, bins[0], bins[-1]), bins=bins, weights=W_Z[mZ])[0]
cRes_cut = np.histogram(np.clip(dfRes[mRes].muon_pt, bins[0], bins[-1]), bins=bins, weights=W_Res[mRes])[0]
print("Efficiencies for %d GeV:\n"%threshold)
print("Higgs : ", np.sum(cH_cut)/np.sum(cH))
print("Data : ", np.sum(cdata_cut)/np.sum(cdata))
print("Z : ", np.sum(cZ_cut)/np.sum(cZ))
print("ggSpin0(200) : ", np.sum(cRes_cut)/np.sum(cRes))

threshold = 12

mdata = dfdata.muon_pt>threshold
mH = dfH.muon_pt>threshold
mZ = dfZ.muon_pt>threshold
mRes = dfRes.muon_pt>threshold
cdata_cut = np.histogram(np.clip(dfdata[mdata].muon_pt, bins[0], bins[-1]), bins=bins)[0]
cH_cut = np.histogram(np.clip(dfH[mH].muon_pt, bins[0], bins[-1]), bins=bins, weights=W_H[mH])[0]
cZ_cut = np.histogram(np.clip(dfZ[mZ].muon_pt, bins[0], bins[-1]), bins=bins, weights=W_Z[mZ])[0]
cRes_cut = np.histogram(np.clip(dfRes[mRes].muon_pt, bins[0], bins[-1]), bins=bins, weights=W_Res[mRes])[0]
print("Efficiencies for %d GeV:\n"%threshold)
print("Higgs : ", np.sum(cH_cut)/np.sum(cH))
print("Data : ", np.sum(cdata_cut)/np.sum(cdata))
print("Z : ", np.sum(cZ_cut)/np.sum(cZ))
print("ggSpin0(200) : ", np.sum(cRes_cut)/np.sum(cRes))

# %%
thresholds = np.linspace(0, 20, 300)
effZ = []
effRes = []
effH = []
effdata = []

for t in thresholds:
    mH = dfH.muon_pt>t
    mZ = dfZ.muon_pt>t
    mRes = dfRes.muon_pt>t
    mdata = dfdata.muon_pt>t
    cdata_cut = np.histogram(np.clip(dfdata[mdata].muon_pt, bins[0], bins[-1]), bins=bins)[0]
    cH_cut = np.histogram(np.clip(dfH[mH].muon_pt, bins[0], bins[-1]), bins=bins, weights=W_H[mH])[0]
    cZ_cut = np.histogram(np.clip(dfZ[mZ].muon_pt, bins[0], bins[-1]), bins=bins, weights=W_Z[mZ])[0]
    cRes_cut = np.histogram(np.clip(dfRes[mRes].muon_pt, bins[0], bins[-1]), bins=bins, weights=W_Res[mRes])[0]

    effH.append(np.sum(cH_cut)/np.sum(cH))
    effdata.append(np.sum(cdata_cut)/np.sum(cdata))
    effZ.append(np.sum(cZ_cut)/np.sum(cZ))
    effRes.append(np.sum(cRes_cut)/np.sum(cRes))
# %%
fig, ax = plt.subplots(1, 1)
ax.plot(thresholds, effH, label='ggH')
ax.plot(thresholds, effZ, label='ZJets')
ax.plot(thresholds, effRes, label='ggSpin0(200)')
ax.plot(thresholds, effdata, label='Data 1A')
ax.set_title("Efficiency drop for Triggering Muon pt threshold")
ax.grid(True)
ax.set_ylim(0.25, 1.01)
ax.set_xlim(0,16 )
ax.legend()
ax.set_xlabel("Muon pt threshold [GeV]")

# %%
fig, ax = plt.subplots(1, 1)
ax.hist(dfRes.muon_pt, bins=np.linspace(0, 50, 101), density=True, weights=W_Res, histtype='step')
ax.hist(dfRes.muon_pt[abs(dfRes.muon_eta)<0.8], bins=np.linspace(0, 50, 101), density=True, weights=W_Res[abs(dfRes.muon_eta)<0.8], histtype='step')

# %%

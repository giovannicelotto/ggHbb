# %%
from functions import cut, cut_advanced, loadMultiParquet_Data_new, loadMultiParquet_v2, getCommonFilters, getDfProcesses_v2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
nn= False
import mplhep as hep
hep.style.use("CMS")
from functions import getDfProcesses_v2

# %%
isHiggsList = [0, 36]
dfProcessesMC = getDfProcesses_v2()[0]
xsections = dfProcessesMC.iloc[isHiggsList].xsection
if nn:
    featuresForTraining = list(np.load("/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/Mar21_2_0p0/featuresForTraining.npy"))
    if "massHypo" in featuresForTraining:
        featuresForTraining.remove("massHypo") 
    else:
        pass
else:
    featuresForTraining = ['dijet_pt', 'jet2_btagDeepFlavB', 'jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'muon_pt','muon_dxySig', 'jet1_pt', 'jet2_pt']
# %%
dfsData_, lumi, fileNumberListData = loadMultiParquet_Data_new(dataTaking=[1, 2], nReals=[100, 500], columns=featuresForTraining+['jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass', 'jet1_eta', 'jet2_eta'], filters=None, returnFileNumberList=True)
dfsMC_, sumw, fileNumberListMC = loadMultiParquet_v2(paths=isHiggsList, nMCs=-1, columns=featuresForTraining+[ 'sf', 'PU_SF','jet1_btag_central', 'genWeight', 'jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass', 'jet1_eta', 'jet2_eta'], returnNumEventsTotal=True, filters=None, returnFileNumberList=True)
# %%
dfsData_f, lumi = loadMultiParquet_Data_new(dataTaking=[1, 2], nReals=[16, 80], columns=featuresForTraining+['jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass'], filters=getCommonFilters(btagTight=False),selectFileNumberList=fileNumberListData )
dfsMC_f, sumw_f = loadMultiParquet_v2(paths=isHiggsList, nMCs=-1, columns=featuresForTraining+[ 'sf', 'PU_SF','jet1_btag_central','jet2_btagDeepFlavB', 'genWeight', 'jet1_btagDeepFlavB','muon_eta', 'dijet_mass'], returnNumEventsTotal=True, filters=getCommonFilters(btagTight=False), selectFileNumberList=fileNumberListMC)
# %%

dfsMC_[0].jet1_btag_central = np.where(dfsMC_[0].jet1_btag_central.values<-9000, 1, dfsMC_[0].jet1_btag_central.values)
dfsMC_[1].jet1_btag_central = np.where(dfsMC_[1].jet1_btag_central.values<-9000, 1, dfsMC_[1].jet1_btag_central.values)


fractionGGH_unfiltered = (dfsMC_[0].sf * dfsMC_[0].jet1_btag_central * dfsMC_[0].PU_SF*dfsMC_[0].genWeight).sum()/sumw[0]
fractionGGH_filtered = (dfsMC_f[0].sf * dfsMC_f[0].jet1_btag_central * dfsMC_f[0].PU_SF*dfsMC_f[0].genWeight).sum()/sumw_f[0]
fractionVBF_unfiltered = (dfsMC_[1].sf * dfsMC_[1].jet1_btag_central * dfsMC_[1].PU_SF*dfsMC_[1].genWeight).sum()/sumw[1]
fractionVBF_filtered = (dfsMC_f[1].sf * dfsMC_f[1].jet1_btag_central * dfsMC_f[1].PU_SF*dfsMC_f[1].genWeight).sum()/sumw_f[1]
print("ggH Filter Preselection Efficiency : %.1f%%"%(fractionGGH_filtered/fractionGGH_unfiltered*100))
print("VBF Filter Preselection Efficiency : %.1f%%"%(fractionVBF_filtered/fractionVBF_unfiltered*100))
print("Data Filter Preselection Efficiency : %.2f%%"%(len(dfsData_f[0])*100/len(dfsData_[0])))

# %%
data = pd.concat(dfsData_)
# %%
maskData = (data.dijet_mass>0) & (data.dijet_mass<np.inf)
maskHiggs = (dfsMC_[0].dijet_mass>0) & (dfsMC_[0].dijet_mass<np.inf)
maskVBF = (dfsMC_[1].dijet_mass>0) & (dfsMC_[1].dijet_mass<np.inf)
fullLumi_factor = 41.6/lumi
totalEventData_ = np.sum(maskData)
totalEventHiggs_ = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF_ = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Jets selected with |eta|<2.5, PU ID and Jet ID for both jets")
print("Only muon in jet1. TrigSF Jet1BTag PUSF GenWEight")
print("Data : %d   |  %d"%(totalEventData_,totalEventData_*fullLumi_factor) )
print("ggh :  %d   |  %d"%(totalEventHiggs_,totalEventHiggs_*fullLumi_factor) )
print("VBF :  %d   |  %d"%(totalEventVBF_,totalEventVBF_*fullLumi_factor) )

# %%


#maskData = (data.dijet_mass>100) & (data.dijet_mass<150)
#maskHiggs = (dfsMC_[0].dijet_mass>100) & (dfsMC_[0].dijet_mass<150)
#maskVBF = (dfsMC_[1].dijet_mass>100) & (dfsMC_[1].dijet_mass<150)
#totalEventData = np.sum(maskData)
#totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs].sum()/sumw_f[0]*xsections.iloc[0]*lumi*1000
#totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF].sum()/sumw_f[1]*xsections.iloc[1]*lumi*1000
#print("100<mjj<150")
#print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
#print("ggh :  %d   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
#print("VBF :  %d   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )



maskData = maskData & (data.jet1_pt>20) & (data.jet2_pt>20)
maskHiggs = maskHiggs & (dfsMC_[0].jet1_pt>20) & (dfsMC_[0].jet2_pt>20)
maskVBF = maskVBF & (dfsMC_[1].jet1_pt>20) & (dfsMC_[1].jet2_pt>20)
totalEventData = np.sum(maskData)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Jet pt > 20")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %d   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %d   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )


maskData = maskData & (data.muon_pt>9) & (abs(data.muon_dxySig)>6) & (abs(data.muon_eta)<1.5)
maskHiggs = maskHiggs  & (dfsMC_[0].muon_pt>9) & (abs(dfsMC_[0].muon_dxySig)>6) & (abs(dfsMC_[0].muon_eta)<1.5)
maskVBF = maskVBF  & (dfsMC_[1].muon_pt>9) & (abs(dfsMC_[1].muon_dxySig)>6) & (abs(dfsMC_[1].muon_eta)<1.5)
totalEventData = np.sum(maskData)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Muon pT > 9 & abs(Muon dxySig)>6 & abs(muon_eta)<1.5")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %d   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %d   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )


maskData = maskData     & (data.jet1_btagDeepFlavB > 0.2783)         & (data.jet2_btagDeepFlavB > 0.2783)
maskHiggs = maskHiggs   & (dfsMC_[0].jet1_btagDeepFlavB > 0.2783)    & (dfsMC_[0].jet2_btagDeepFlavB > 0.2783)
maskVBF = maskVBF       & (dfsMC_[1].jet1_btagDeepFlavB > 0.2783)    & (dfsMC_[1].jet2_btagDeepFlavB > 0.2783)       
totalEventData = np.sum(maskData)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Btag > M")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %d   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %d   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )



maskData = maskData     & (data.jet1_mass>0) & (data.jet2_mass>0)
maskHiggs = maskHiggs   & (dfsMC_[0].jet1_mass>0) & (dfsMC_[0].jet2_mass>0)
maskVBF = maskVBF       & (dfsMC_[1].jet1_mass>0) & (dfsMC_[1].jet2_mass>0)
totalEventData = np.sum(maskData)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Jet_mass > 0")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %.1f   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %.1f   |  %d\n\n\n"%(totalEventVBF,totalEventVBF*fullLumi_factor) )

print("\n\nBefore categories Events:")
print("ggH\t", totalEventHiggs)
print("VBF\t", totalEventVBF)
ggHafterCategories = 0
VBFafterCategories = 0
# ----------------------------
# --- Cat 
# ----------------------------
maskData_cat2 = maskData     & (data.dijet_pt>=160) & (data.jet1_btagDeepFlavB > 0.71) & (data.jet2_btagDeepFlavB > 0.71)
maskHiggs_cat2 = maskHiggs   & (dfsMC_[0].dijet_pt>=160) & (dfsMC_[0].jet1_btagDeepFlavB > 0.71) & (dfsMC_[0].jet2_btagDeepFlavB > 0.71)
maskVBF_cat2 = maskVBF       & (dfsMC_[1].dijet_pt>=160) & (dfsMC_[1].jet1_btagDeepFlavB > 0.71) & (dfsMC_[1].jet2_btagDeepFlavB > 0.71)
totalEventData = np.sum(maskData_cat2)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs_cat2].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF_cat2].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Pt 160-Inf btag > T")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %.2f   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %.2f   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )
print("Significance %.2f\n" %((totalEventHiggs+totalEventVBF)/np.sqrt(totalEventData)*np.sqrt(fullLumi_factor)))
ggHafterCategories=ggHafterCategories+totalEventHiggs
VBFafterCategories=VBFafterCategories+totalEventVBF



maskData_cat1 = maskData     & (data.dijet_pt>=100) & (data.dijet_pt<160) & (data.jet1_btagDeepFlavB > 0.71) & (data.jet2_btagDeepFlavB > 0.71)
maskHiggs_cat1 = maskHiggs   & (dfsMC_[0].dijet_pt>=100) & (dfsMC_[0].dijet_pt<160) & (dfsMC_[0].jet1_btagDeepFlavB > 0.71) & (dfsMC_[0].jet2_btagDeepFlavB > 0.71)
maskVBF_cat1 = maskVBF       & (dfsMC_[1].dijet_pt>=100) & (dfsMC_[1].dijet_pt<160) & (dfsMC_[1].jet1_btagDeepFlavB > 0.71) & (dfsMC_[1].jet2_btagDeepFlavB > 0.71)
totalEventData = np.sum(maskData_cat1)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs_cat1].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF_cat1].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Pt 100-160 btag>T")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %.2f   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %.2f   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )
print("Significance %.2f\n" %((totalEventHiggs+totalEventVBF)/np.sqrt(totalEventData)*np.sqrt(fullLumi_factor)))
ggHafterCategories=ggHafterCategories+totalEventHiggs
VBFafterCategories=VBFafterCategories+totalEventVBF

maskData_cat0 = maskData     & (data.dijet_pt<100)
maskHiggs_cat0 = maskHiggs   & (dfsMC_[0].dijet_pt<100)
maskVBF_cat0 = maskVBF       & (dfsMC_[1].dijet_pt<100)
totalEventData = np.sum(maskData_cat0)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs_cat0].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF_cat0].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Pt<100")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %.2f   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %.2f   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )
print("Significance %.2f\n" %((totalEventHiggs+totalEventVBF)/np.sqrt(totalEventData)*np.sqrt(fullLumi_factor)))
ggHafterCategories=ggHafterCategories+totalEventHiggs
VBFafterCategories=VBFafterCategories+totalEventVBF

maskData_catLost = maskData     & (data.dijet_pt>=100) & (data.dijet_pt<160) & ~((data.jet1_btagDeepFlavB>0.71) & (data.jet2_btagDeepFlavB>0.71))
maskHiggs_catLost = maskHiggs   & (dfsMC_[0].dijet_pt>=100) & (dfsMC_[0].dijet_pt<160) & ~((dfsMC_[0].jet1_btagDeepFlavB>0.71) & (dfsMC_[0].jet2_btagDeepFlavB>0.71))
maskVBF_catLost = maskVBF       & (dfsMC_[1].dijet_pt>=100) & (dfsMC_[1].dijet_pt<160) & ~((dfsMC_[1].jet1_btagDeepFlavB>0.71) & (dfsMC_[1].jet2_btagDeepFlavB>0.71) )
totalEventData = np.sum(maskData_catLost)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs_catLost].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF_catLost].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Cat lost1")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %.2f   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %.2f   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )
print("Significance %.2f\n" %((totalEventHiggs+totalEventVBF)/np.sqrt(totalEventData)*np.sqrt(fullLumi_factor)))
ggHafterCategories=ggHafterCategories+totalEventHiggs
VBFafterCategories=VBFafterCategories+totalEventVBF

maskData_catLost = maskData     & (data.dijet_pt>=160)  & ~((data.jet1_btagDeepFlavB>0.71) & (data.jet2_btagDeepFlavB>0.71))
maskHiggs_catLost = maskHiggs   & (dfsMC_[0].dijet_pt>=160)  & ~((dfsMC_[0].jet1_btagDeepFlavB>0.71) & (dfsMC_[0].jet2_btagDeepFlavB>0.71))
maskVBF_catLost = maskVBF       & (dfsMC_[1].dijet_pt>=160)  & ~((dfsMC_[1].jet1_btagDeepFlavB>0.71) & (dfsMC_[1].jet2_btagDeepFlavB>0.71) )
totalEventData = np.sum(maskData_catLost)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs_catLost].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF_catLost].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Cat lost2")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %.2f   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %.2f   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )
print("Significance %.2f\n" %((totalEventHiggs+totalEventVBF)/np.sqrt(totalEventData)*np.sqrt(fullLumi_factor)))
ggHafterCategories=ggHafterCategories+totalEventHiggs
VBFafterCategories=VBFafterCategories+totalEventVBF

print("\nAfter categories Events:")
print("ggH\t", ggHafterCategories)
print("VBF\t", VBFafterCategories)

# %%
#fig, ax = plt.subplots(1, 1)
#ax.hist(data.dijet_mass[maskData_catLost], bins=np.linspace(40, 300,101))
# %%
preselectionSteps = [
    "Initial", "Jet pt > 20", "Muon cuts",
    "Btag > M", "40 < Dijet Mass < 300",
]

mydfs = [pd.concat(dfsData_),dfsMC_[0], dfsMC_[1] ]
counts={
    'Data':[],
    'ggH':[],
    'VBF':[]
}

counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)



mydfs = cut(mydfs, 'jet1_pt', 20, None)
mydfs = cut(mydfs, 'jet2_pt', 20, None)
mydfs = cut(mydfs, 'jet1_eta', -2.5, 2.5)
mydfs = cut(mydfs, 'jet2_eta', -2.5, 2.5)
counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

for idx, df in enumerate(mydfs):
    m = (df.muon_pt>=9) & (abs(df.muon_eta)<=1.5) & (abs(df.muon_dxySig)>=6)
    mydfs[idx] = df[m]
counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)


mydfs = cut(mydfs, 'jet1_btagDeepFlavB', 0.2783, None)
mydfs = cut(mydfs, 'jet2_btagDeepFlavB', 0.2783, None)
counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

#mydfs = cut(mydfs, 'jet1_mass', 0, None)
#mydfs = cut(mydfs, 'jet2_mass', 0, None)
#counts['Data'].append(len(mydfs[0]))
#counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
#counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

mydfs = cut(mydfs, 'dijet_mass', 40, 300)
counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)




# Categories in pT

mydfs_0 = cut(mydfs, 'dijet_pt', None, 100-1e-10)
counts['Data'].append(len(mydfs_0[0]))
counts['ggH'].append((mydfs_0[1].sf*mydfs_0[1].PU_SF*mydfs_0[1].jet1_btag_central*mydfs_0[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_0[2].sf*mydfs_0[2].PU_SF*mydfs_0[2].jet1_btag_central*mydfs_0[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

mydfs_1 = cut(mydfs, 'dijet_pt', 100-1e-10, 160-1e-10)
mydfs_1 = cut(mydfs_1, 'jet1_btagDeepFlavB', 0.71, None)
mydfs_1 = cut(mydfs_1, 'jet2_btagDeepFlavB', 0.71, None)
counts['Data'].append(len(mydfs_1[0]))
counts['ggH'].append((mydfs_1[1].sf*mydfs_1[1].PU_SF*mydfs_1[1].jet1_btag_central*mydfs_1[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_1[2].sf*mydfs_1[2].PU_SF*mydfs_1[2].jet1_btag_central*mydfs_1[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

mydfs_2 = cut(mydfs, 'dijet_pt', 160-1e-10, None)
mydfs_2 = cut(mydfs_2, 'jet1_btagDeepFlavB', 0.71, None)
mydfs_2 = cut(mydfs_2, 'jet2_btagDeepFlavB', 0.71, None)
counts['Data'].append(len(mydfs_2[0]))
counts['ggH'].append((mydfs_2[1].sf*mydfs_2[1].PU_SF*mydfs_2[1].jet1_btag_central*mydfs_2[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_2[2].sf*mydfs_2[2].PU_SF*mydfs_2[2].jet1_btag_central*mydfs_2[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

mydfs_lost_1 = cut(mydfs, 'dijet_pt', 100-1e-10, 160-1e-10)
for idx, df in enumerate(mydfs_lost_1):
    m = ~((df.jet1_btagDeepFlavB>0.71) & (df.jet2_btagDeepFlavB>0.71))
    mydfs_lost_1[idx] = df[m]

counts['Data'].append(len(mydfs_lost_1[0]))
counts['ggH'].append((mydfs_lost_1[1].sf*mydfs_lost_1[1].PU_SF*mydfs_lost_1[1].jet1_btag_central*mydfs_lost_1[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_lost_1[2].sf*mydfs_lost_1[2].PU_SF*mydfs_lost_1[2].jet1_btag_central*mydfs_lost_1[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

mydfs_lost_2 = cut(mydfs, 'dijet_pt', 160-1e-10, None)
for idx, df in enumerate(mydfs_lost_2):
    m = ~((df.jet1_btagDeepFlavB>0.71) & (df.jet2_btagDeepFlavB>0.71))
    mydfs_lost_2[idx] = df[m]

counts['Data'].append(len(mydfs_lost_2[0]))
counts['ggH'].append((mydfs_lost_2[1].sf*mydfs_lost_2[1].PU_SF*mydfs_lost_2[1].jet1_btag_central*mydfs_lost_2[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_lost_2[2].sf*mydfs_lost_2[2].PU_SF*mydfs_lost_2[2].jet1_btag_central*mydfs_lost_2[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)
# %%
for process in counts.keys():
    counts[process]=np.array(counts[process])/(counts[process][0])*100


# %%
fig, ax = plt.subplots(figsize=(30, 6))
x = np.arange(len(preselectionSteps))

# Bar width
width = 0.25
bars_data = ax.bar(x[:len(preselectionSteps)] - width, counts['Data'][:len(preselectionSteps)], width, label='Data', color='black', alpha=0.7)
bars_ggH = ax.bar(x[:len(preselectionSteps)], counts['ggH'][:len(preselectionSteps)], width, label='ggH', color='red', alpha=0.7)
bars_vbf = ax.bar(x[:len(preselectionSteps)] + width, counts['VBF'][:len(preselectionSteps)], width, label='VBF', color='blue', alpha=0.7)


def format_number(value):
    if value >= 1e6:
        return f'{value / 1e6:.1f}M'
    elif value >= 1e3:
        return f'{value / 1e3:.1f}K'
    else:
        return str(round(value))
    

def format_number_percentage(value):
    if value >99:
        return "%.1f"%value
    elif value >10:
        return "%.2f"%value
    elif 0.1<value <10:
        return "%.2f"%value
    else:
        return "%.3f"%value

# Add formatted integer values on top of the bars
for bars in [bars_data, bars_ggH, bars_vbf]:
    for bar in bars:
        height = bar.get_height()
        formatted_value = format_number_percentage(height)
        ax.text(bar.get_x() + bar.get_width() / 2, height, formatted_value, ha='center', va='bottom', fontsize=30)


# Labels and title
ax.set_xlabel("Selection Steps", fontsize=30)
ax.set_ylabel("Efficiency [%]", fontsize=30)
ax.set_xticks(x[:len(preselectionSteps)])
#ax.set_yscale('log')
ax.set_xticklabels(preselectionSteps[:len(preselectionSteps)], rotation=0, ha="center", fontsize=30)
ax.legend(fontsize=30, bbox_to_anchor=(1, 1))
ax.tick_params(labelsize=30)
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()







# %%


categoriesLabels = ["Dijet pT < 100", "100 < Dijet pT < 160", "Dijet pT > 160", "Dijet pT > 160 (noTight)", "100 < Dijet pT < 160 (noTight)"]

ggH_fractions = np.array(counts['ggH'][-len(categoriesLabels):])   # / sum(counts['ggH'][-len(categoriesLabels):]) * 100
vbf_fractions = np.array(counts['VBF'][-len(categoriesLabels):])   # / sum(counts['VBF'][-len(categoriesLabels):]) * 100
data_fractions = np.array(counts['Data'][-len(categoriesLabels):]) #/ sum(counts['Data'][-len(categoriesLabels):]) * 100

# Create figure and axes
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Create bar chart for Data
bars = ax[0].bar(categoriesLabels, data_fractions,color='black')#[['blue', 'green', 'red', 'purple', 'orange'])
#ax[0].set_title('Data Distribution')
ax[0].set_ylabel('Percentage')
ax[0].set_yscale('log')
ax[0].set_ylim(0.001,10000)
ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1]*2)
ax[0].set_xticklabels(categoriesLabels, rotation=45, ha='right')
for bar, value in zip(bars, data_fractions):
    ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_number_percentage(value), ha='center', va='bottom')



# Create bar chart for ggH
bars = ax[1].bar(categoriesLabels, ggH_fractions, color='red')#[['blue', 'green', 'red', 'purple', 'orange'])
#ax[1].set_title('ggH Distribution')
ax[1].set_ylabel('Percentage')
ax[1].set_yscale('log')
ax[1].set_ylim(0.001,10000)
ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1]*2)
ax[1].set_xticklabels(categoriesLabels, rotation=45, ha='right')
for bar, value in zip(bars, ggH_fractions):
    ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_number_percentage(value), ha='center', va='bottom')


# Create bar chart for VBF
bars = ax[2].bar(categoriesLabels, vbf_fractions, color='blue')#['blue', 'green', 'red', 'purple', 'orange'])
#ax[2].set_title('VBF Distribution')
ax[2].set_ylabel('Percentage')
ax[2].set_yscale('log')
ax[2].set_ylim(0.001,10000)
ax[2].set_ylim(ax[2].get_ylim()[0], ax[2].get_ylim()[1]*2)
ax[2].set_xticklabels(categoriesLabels, rotation=45, ha='right')
for bar, value in zip(bars, vbf_fractions):
    ax[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_number_percentage(value), ha='center', va='bottom')


plt.tight_layout()
plt.show()




# %%
# Check after cut that filtered and unfilted + cuts is the same
dfsData_f =cut(dfsData_f, 'dijet_mass', 40, 300)
dfsMC_f =cut(dfsMC_f, 'dijet_mass', 40, 300)
assert len(mydfs[0])==len(dfsData_f[0])
assert len(mydfs[1])==len(dfsMC_f[0])
assert len(mydfs[2])==len(dfsMC_f[1])
assert len(mydfs[1])== len(mydfs_0[1]) + len(mydfs_1[1]) + len(mydfs_2[1]) + len(mydfs_lost_1[1]) + len(mydfs_lost_2[1]) 
assert len(mydfs[0])== len(mydfs_0[0]) + len(mydfs_1[0]) + len(mydfs_2[0]) + len(mydfs_lost_1[0]) + len(mydfs_lost_2[0]) 
assert len(mydfs[0])== len(mydfs_0[0]) + len(mydfs_1[0]) + len(mydfs_2[0]) + len(mydfs_lost_1[0]) + len(mydfs_lost_2[0]) 
print("Preselection Cuts are consistent!!")



# %%




# Neural Networks
# NN1 score

# %%
if nn:
    import torch, sys
    sys.path.append("/t3home/gcelotto/ggHbb/PNN")
    #outFolder="/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/Feb21_0p0"
    modelName = "Mar21_2_0p0"
    modelDir="/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s"%modelName
    modelName=modelDir+"/model/model.pth"
    model2 = torch.load(modelName, map_location=torch.device('cpu'), weights_only=False)
    model2.eval()

    modelName = "Mar21_1_0p0"
    modelDir="/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s"%modelName
    modelName=modelDir+"/model/model.pth"
    model1 = torch.load(modelName, map_location=torch.device('cpu'), weights_only=False)
    model1.eval()
# %%
for df in mydfs:
    conditions = [
        (df['dijet_pt'] < 100),   # Condition for 0
        (df['dijet_pt'] >= 100) & (df['dijet_pt'] <= 160),  # Condition for 1
        (df['dijet_pt'] > 160)   # Condition for 2
    ]
    
    choices = [0, 1, 2]  # Values corresponding to conditions
    
    df['cat'] = np.select(conditions, choices, default=-1)  # Default -1 for safety
# %%
#mHiggs = ((dfH.jet1_btagDeepFlavB > 0.71) & (dfH.jet2_btagDeepFlavB < 0.71)) | ((dfH.jet1_btagDeepFlavB < 0.71) & (dfH.jet2_btagDeepFlavB > 0.71)) | ((dfH.jet1_btagDeepFlavB < 0.71) & (dfH.jet2_btagDeepFlavB < 0.71))
#mData = ((dfsData[0].jet1_btagDeepFlavB > 0.71) & (dfsData[0].jet2_btagDeepFlavB < 0.71)) | ((dfsData[0].jet1_btagDeepFlavB < 0.71) & (dfsData[0].jet2_btagDeepFlavB > 0.71)) | ((dfsData[0].jet1_btagDeepFlavB < 0.71) & (dfsData[0].jet2_btagDeepFlavB < 0.71))
#dfH = dfH[mHiggs]
#dfsData[0] = dfsData[0][mData]
# %%
if nn:    
    from helpers.scaleUnscale import scale, unscale
    modelName = "Mar21_1_0p0"
    #featuresForTraining.remove(['dijet_mass'])
    modelDir="/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s"%modelName
    mydfs_1[0]  = scale(mydfs[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" ,fit=False)
    mydfs_1[1]  = scale(mydfs[1], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" ,fit=False)
    mydfs_1[2]  = scale(mydfs[2], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" ,fit=False)
#
    data_tensor = torch.tensor(np.float32(mydfs_1[1][featuresForTraining].values)).float()
    mydfs_1[1]['NN1'] = model1(data_tensor).detach().numpy()
    data_tensor = torch.tensor(np.float32(mydfs_1[0][featuresForTraining].values)).float()
    mydfs_1[0]['NN1'] = model1(data_tensor).detach().numpy()
    data_tensor = torch.tensor(np.float32(mydfs_1[2][featuresForTraining].values)).float()
    mydfs_1[2]['NN1'] = model1(data_tensor).detach().numpy()


    mydfs_1[0]  = unscale(mydfs_1[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" )
    mydfs_1[1]  = unscale(mydfs_1[1], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" )
    mydfs_1[2]  = unscale(mydfs_1[2], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" )


    modelName = "Mar21_2_0p0"
    modelDir="/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s"%modelName
    mydfs_2[0]  = scale(mydfs[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" ,fit=False)
    mydfs_2[1]  = scale(mydfs[1], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" ,fit=False)
    mydfs_2[2]  = scale(mydfs[2], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" ,fit=False)

    data_tensor = torch.tensor(np.float32(mydfs_2[1][featuresForTraining].values)).float()
    mydfs_2[1]['NN2'] = model2(data_tensor).detach().numpy()
    data_tensor = torch.tensor(np.float32(mydfs_2[0][featuresForTraining].values)).float()
    mydfs_2[0]['NN2'] = model2(data_tensor).detach().numpy()
    data_tensor = torch.tensor(np.float32(mydfs_2[2][featuresForTraining].values)).float()
    mydfs_2[2]['NN2'] = model2(data_tensor).detach().numpy()

    mydfs_2[0]  = unscale(mydfs_2[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" )
    mydfs_2[1]  = unscale(mydfs_2[1], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" )
    mydfs_2[2]  = unscale(mydfs_2[2], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" )

# %%

fig, ax = plt.subplots(1, 1)
bins=np.linspace(0, 1, 21)


ax.hist(
    mydfs_1[0].NN1[(mydfs_1[0].cat == 1)],
    bins=bins,
    histtype='step', color='black', linewidth=2,
    weights=np.ones_like(mydfs_1[0].NN1[(mydfs_1[0].cat == 1)]) / 
            len(mydfs_1[0].NN1[(mydfs_1[0].cat == 1)]), label='Data'
)
ax.hist(
    mydfs_1[1].NN1[(mydfs_1[1].cat == 1)],
    bins=bins,
    histtype='step', color='red', linewidth=2,
    weights=np.ones_like(mydfs_1[1].NN1[(mydfs_1[1].cat == 1)]) / 
            len(mydfs_1[1].NN1[(mydfs_1[1].cat == 1)]), label='ggH'
)
ax.hist(
    mydfs_1[2].NN1[(mydfs_1[2].cat == 1)],
    bins=bins,
    histtype='step', color='blue', linewidth=2,
    weights=np.ones_like(mydfs_1[2].NN1[(mydfs_1[2].cat == 1)]) / 
            len(mydfs_1[2].NN1[(mydfs_1[2].cat == 1)]), label='VBF'
)

ax.legend()
ax.set_xlabel("NN Cat1 score")
ax.legend()





fig, ax = plt.subplots(1, 1)
bins=np.linspace(0, 1, 21)


ax.hist(
    mydfs_2[0].NN2[(mydfs_2[0].cat == 2)],
    bins=bins,
    histtype='step', color='black', linewidth=2,
    weights=np.ones_like(mydfs_2[0].NN2[(mydfs_2[0].cat == 2)]) / 
            len(mydfs_2[0].NN2[(mydfs_2[0].cat == 2)]), label='Data'
)
ax.hist(
    mydfs_2[1].NN2[(mydfs_2[1].cat == 2)],
    bins=bins,
    histtype='step', color='red', linewidth=2,
    weights=np.ones_like(mydfs_2[1].NN2[(mydfs_2[1].cat == 2)]) / 
            len(mydfs_2[1].NN2[(mydfs_2[1].cat == 2)]), label='ggH'
)
ax.hist(
    mydfs_2[2].NN2[(mydfs_2[2].cat == 2)],
    bins=bins,
    histtype='step', color='blue', linewidth=2,
    weights=np.ones_like(mydfs_2[2].NN2[(mydfs_2[2].cat == 2)]) / 
            len(mydfs_2[2].NN2[(mydfs_2[2].cat == 2)]), label='VBF'
)

ax.legend()
ax.set_xlabel("NN Cat2-3 score")
ax.legend()
# %%


best_threshold = None
best_significance = -np.inf  # Start with a very low value

thresholds = np.linspace(0.0, 0.9, 400)  # Try values from 0.0 to 1.0

for thresh in thresholds:
    ranges = [(thresh, 1)]  # Vary the lower bound
    efficiencies = {}

    for i, label in zip(range(3), ['Data', 'ggH']):
        data = mydfs_1[i].NN1[(mydfs_1[i].cat == 1)]
        weights = np.ones_like(data) / len(data)  # Normalize histogram
        
        for r in ranges:
            efficiencies[label] = weights[(data >= r[0]) & (data < r[1])].sum()
    
    # Compute significance metric
    if efficiencies['Data'] > 0:  # Avoid division by zero
        significance = (efficiencies['ggH']) / np.sqrt(efficiencies['Data'])
        
        # Check if this is the best significance so far
        if significance > best_significance:
            best_significance = significance
            best_threshold = thresh

print(f"Optimal threshold: {best_threshold:.3f}, Maximum significance: {best_significance:.4f}")
# %%
ranges = [(best_threshold, 1)]
for i, label in zip(range(3), ['Data', 'ggH', 'VBF']):
    data = mydfs_1[i].NN1[(mydfs_1[i].cat == 1)]
    weights = np.ones_like(data) / len(data)  # Normalize histogram
    
    for r in ranges:
        area = weights[(data >= r[0]) & (data < r[1])].sum()
        print(f"Efficiency for {label} in range {r}: {area:.4f}")

# %%
import numpy as np

best_cuts = None
best_significance = -np.inf  # Start with a low value

low_cut_values = np.linspace(0.0, 0.6, 100)  # Test lower cuts from 0.3 to 0.55
high_cut_values = np.linspace(0.4, 1.0, 100)  # Test upper cuts from 0.6 to 1.0

for low_cut in low_cut_values:
    for high_cut in high_cut_values:
        if low_cut >= high_cut:
            continue  # Ensure a valid separation

        ranges = [(low_cut, high_cut), (high_cut, 1)]  # Mid-purity and high-purity
        efficiencies = {'Data': [], 'ggH': []}

        for i, label in zip(range(3), ['Data', 'ggH']):
            data = mydfs_2[i].NN2[(mydfs_2[i].cat == 2)]
            weights = np.ones_like(data) / len(data)  # Normalize histogram

            for r in ranges:
                efficiencies[label].append(weights[(data >= r[0]) & (data < r[1])].sum())

        # Compute significance
        if efficiencies['Data'][0] > 0 and efficiencies['Data'][1] > 0:  # Avoid division by zero
            significance = np.sqrt(
                (efficiencies['ggH'][0] / np.sqrt(efficiencies['Data'][0])) ** 2 +
                (efficiencies['ggH'][1] / np.sqrt(efficiencies['Data'][1])) ** 2
            )

            # Track best cuts
            if significance > best_significance:
                best_significance = significance
                best_cuts = (low_cut, high_cut)

print(f"Optimal cuts: Low purity < {best_cuts[0]:.3f}, High purity > {best_cuts[1]:.3f}")
print(f"Maximum significance: {best_significance:.4f}")

# %%
print("\n\n\n\n")
ranges = [(best_cuts[0], best_cuts[1]), (best_cuts[1], 1)]
for i, label in zip(range(3), ['Data', 'ggH', 'VBF']):
    data = mydfs_2[i].NN2[(mydfs_2[i].cat == 2)]
    weights = np.ones_like(data) / len(data)  # Normalize histogram
    
    for r in ranges:
        area = weights[(data >= r[0]) & (data < r[1])].sum()
        print(f"Efficiency for {label} in range {r}: {area:.4f}")






# %%
mydfs_1 = cut(mydfs_1, 'dijet_pt', 100-1e-12, 160)
mydfs_1 = cut(mydfs_1, 'NN1', best_threshold,None)
mydfs_1 = cut(mydfs_1, 'jet1_btagDeepFlavB', 0.71, None)
mydfs_1 = cut(mydfs_1, 'jet2_btagDeepFlavB', 0.71, None)
counts["Data"]=list(counts["Data"])
counts["ggH"]=list(counts["ggH"])
counts["VBF"]=list(counts["VBF"])

counts['Data'].append(len(mydfs_1[0]))
counts['ggH'].append((mydfs_1[1].sf*mydfs_1[1].PU_SF*mydfs_1[1].jet1_btag_central*mydfs_1[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*lumi*1000)
counts['VBF'].append((mydfs_1[2].sf*mydfs_1[2].PU_SF*mydfs_1[2].jet1_btag_central*mydfs_1[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*lumi*1000)


mydfs_2p = cut(mydfs_2, 'dijet_pt', 160-1e-6, None)
mydfs_2p = cut(mydfs_2p, 'NN2', best_cuts[0], best_cuts[1])
mydfs_2p = cut(mydfs_2p, 'jet1_btagDeepFlavB', 0.71, None)
mydfs_2p = cut(mydfs_2p, 'jet2_btagDeepFlavB', 0.71, None)
counts['Data'].append(len(mydfs_2p[0]))
counts['ggH'].append((mydfs_2p[1].sf*mydfs_2p[1].PU_SF*mydfs_2p[1].jet1_btag_central*mydfs_2p[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*lumi*1000)
counts['VBF'].append((mydfs_2p[2].sf*mydfs_2p[2].PU_SF*mydfs_2p[2].jet1_btag_central*mydfs_2p[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*lumi*1000)


mydfs_3 = cut(mydfs_2, 'dijet_pt', 160-1e-6, None)
mydfs_3 = cut(mydfs_3, 'NN2', best_cuts[1],None)
mydfs_3 = cut(mydfs_3, 'jet1_btagDeepFlavB', 0.71, None)
mydfs_3 = cut(mydfs_3, 'jet2_btagDeepFlavB', 0.71, None)
counts['Data'].append(len(mydfs_3[0]))
counts['ggH'].append((mydfs_3[1].sf*mydfs_3[1].PU_SF*mydfs_3[1].jet1_btag_central*mydfs_3[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*lumi*1000)
counts['VBF'].append((mydfs_3[2].sf*mydfs_3[2].PU_SF*mydfs_3[2].jet1_btag_central*mydfs_3[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*lumi*1000)

# %%

categories=['Cat1', 'Cat2', 'Cat3']
x = np.arange(len(categories))
counts['Data'][-3:]=np.array(counts['Data'][-3:])/totalEventData_*100
counts['ggH'][-3:]=np.array(counts['ggH'][-3:])/totalEventHiggs_*100
counts['VBF'][-3:]=np.array(counts['VBF'][-3:])/totalEventVBF_*100
# %%
fig, ax = plt.subplots(figsize=(30, 6))
# Bar width
width = 0.25
bars_data = ax.bar(x - width, counts['Data'][-3:], width, label='Data', color='black', alpha=0.7)
bars_ggH = ax.bar(x, counts['ggH'][-3:], width, label='ggH', color='red', alpha=0.7)
bars_vbf = ax.bar(x + width, counts['VBF'][-3:], width, label='VBF', color='blue', alpha=0.7)


def format_number(value):
    if value >= 1e6:
        return f'{value / 1e6:.1f}M'
    elif value >= 1e3:
        return f'{value / 1e3:.1f}K'
    else:
        return str(round(value))
    

def format_number_percentage(value):
    if value >10:
        return "%.1f"%value
    elif 0.1<value <10:
        return "%.3f"%value
    elif 0.001<value <0.1:
        return "%.3f"%value
    else:
        return "%.5f"%value

# Add formatted integer values on top of the bars
for bars in [bars_data, bars_ggH, bars_vbf]:
    for bar in bars:
        height = bar.get_height()
        formatted_value = format_number_percentage(height)
        ax.text(bar.get_x() + bar.get_width() / 2, height, formatted_value, ha='center', va='bottom', fontsize=24)


# Labels and title
ax.set_xlabel("Selection Steps", fontsize=30)
ax.set_ylabel("Efficiency [%]", fontsize=30)
ax.set_xticks(x[:5])
ax.set_yscale('log')
ax.set_xticklabels(categories[:5], rotation=0, ha="right", fontsize=30)
ax.legend(fontsize=30, bbox_to_anchor=(1, 1))
ax.tick_params(labelsize=30)
ax.set_ylim(0, 110)
plt.tight_layout()
plt.show()
# %%




finalCat=["Cat1", "Cat2", "Cat3"]





ggH_fractions = np.array(counts['ggH'][-3:])   
vbf_fractions = np.array(counts['VBF'][-3:])   
data_fractions = np.array(counts['Data'][-3:]) 

# Create figure and axes
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Create bar chart for Data
bars = ax[0].bar(finalCat[-3:], data_fractions,color='black')#[['blue', 'green', 'red', 'purple', 'orange'])
#ax[0].set_title('Data Distribution')
ax[0].set_ylabel('Percentage')
ax[0].set_yscale('log')
ax[0].set_ylim(0.001,10000)
ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1]*2)
ax[0].set_xticklabels(finalCat[-3:], rotation=45, ha='right')
for bar, value in zip(bars, data_fractions):
    ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_number_percentage(value), ha='center', va='bottom')



# Create bar chart for ggH
bars = ax[1].bar(finalCat[-3:], ggH_fractions, color='red')#[['blue', 'green', 'red', 'purple', 'orange'])
#ax[1].set_title('ggH Distribution')
ax[1].set_ylabel('Percentage')
ax[1].set_yscale('log')
ax[1].set_ylim(0.001,10000)
ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1]*2)
ax[1].set_xticklabels(finalCat[-3:], rotation=45, ha='right')
for bar, value in zip(bars, ggH_fractions):
    ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_number_percentage(value), ha='center', va='bottom')


# Create bar chart for VBF
bars = ax[2].bar(finalCat[-3:], vbf_fractions, color='blue')#['blue', 'green', 'red', 'purple', 'orange'])
#ax[2].set_title('VBF Distribution')
ax[2].set_ylabel('Percentage')
ax[2].set_yscale('log')
ax[2].set_ylim(0.001,10000)
ax[2].set_ylim(ax[2].get_ylim()[0], ax[2].get_ylim()[1]*2)
ax[2].set_xticklabels(finalCat[-3:], rotation=45, ha='right')
for bar, value in zip(bars, vbf_fractions):
    ax[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_number_percentage(value), ha='center', va='bottom')


plt.tight_layout()
plt.show()




# %%

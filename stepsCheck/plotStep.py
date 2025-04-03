# %%
from functions import cut, cut_advanced, loadMultiParquet_Data_new, loadMultiParquet_v2, getCommonFilters, getDfProcesses_v2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
nn= False
import mplhep as hep
hep.style.use("CMS")
from functions import getDfProcesses_v2
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
isHiggsList = [43, 36]
dfProcessesMC = getDfProcesses_v2()[0]
xsections = dfProcessesMC.iloc[isHiggsList].xsection

featuresForTraining = ['dijet_pt', 'jet1_mass', 'jet2_mass', 'jet1_eta', 'jet2_eta', 'muon_pt','muon_dxySig', 'jet1_pt', 'jet2_pt']
# %%
dfsData_, lumi, fileNumberListData = loadMultiParquet_Data_new(dataTaking=[1, 2], nReals=[10, 50], columns=featuresForTraining+['jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass'], filters=None, returnFileNumberList=True)
dfsMC_, sumw, fileNumberListMC = loadMultiParquet_v2(paths=isHiggsList, nMCs=-1, columns=featuresForTraining+[ 'sf', 'PU_SF','jet1_btag_central', 'genWeight', 'jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass'], returnNumEventsTotal=True, filters=None, returnFileNumberList=True)
# %%
dfsMC_[0].jet1_btag_central = np.where(dfsMC_[0].jet1_btag_central.values<-9000, 1, dfsMC_[0].jet1_btag_central.values)
dfsMC_[1].jet1_btag_central = np.where(dfsMC_[1].jet1_btag_central.values<-9000, 1, dfsMC_[1].jet1_btag_central.values)

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
maskData = maskData & (data.jet1_pt>20) & (data.jet2_pt>20) &(abs(data.jet1_eta)<2.5) & (abs(data.jet2_eta)<2.5)
maskHiggs = maskHiggs & (dfsMC_[0].jet1_pt>20) & (dfsMC_[0].jet2_pt>20) &(abs(dfsMC_[0].jet1_eta)<2.5) & (abs(dfsMC_[0].jet2_eta)<2.5)
maskVBF = maskVBF & (dfsMC_[1].jet1_pt>20) & (dfsMC_[1].jet2_pt>20) &(abs(dfsMC_[1].jet1_eta)<2.5) & (abs(dfsMC_[1].jet2_eta)<2.5)
totalEventData = np.sum(maskData)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Jet pt > 20")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %d   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %d   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )
# %%
maskData = maskData & (data.muon_pt>=9) & (abs(data.muon_dxySig)>=6) & (abs(data.muon_eta)<=1.5)
maskHiggs = maskHiggs  & (dfsMC_[0].muon_pt>=9) & (abs(dfsMC_[0].muon_dxySig)>=6) & (abs(dfsMC_[0].muon_eta)<=1.5)
maskVBF = maskVBF  & (dfsMC_[1].muon_pt>=9) & (abs(dfsMC_[1].muon_dxySig)>=6) & (abs(dfsMC_[1].muon_eta)<=1.5)
totalEventData = np.sum(maskData)
totalEventHiggs = (dfsMC_[0].sf*dfsMC_[0].PU_SF* dfsMC_[0].jet1_btag_central * dfsMC_[0].genWeight)[maskHiggs].sum()/sumw[0]*xsections.iloc[0]*lumi*1000
totalEventVBF = (dfsMC_[1].sf*dfsMC_[1].PU_SF*dfsMC_[1].jet1_btag_central * dfsMC_[1].genWeight)[maskVBF].sum()/sumw[1]*xsections.iloc[1]*lumi*1000
print("Muon pT > 9 & abs(Muon dxySig)>6 & abs(muon_eta)<1.5")
print("Data : %d   |  %d"%(totalEventData,totalEventData*fullLumi_factor) )
print("ggh :  %d   |  %d"%(totalEventHiggs,totalEventHiggs*fullLumi_factor) )
print("VBF :  %d   |  %d"%(totalEventVBF,totalEventVBF*fullLumi_factor) )
# %%

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

# %%

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


# %%
preselectionSteps = [
    "Initial", "Jet pt > 20", "Muon cuts",
    "Btag > M", "40 < Dijet Mass < 300",
]

mydfs = [data,dfsMC_[0], dfsMC_[1] ]
# %%
counts={
    'Data':[],
    'ggH':[],
    'VBF':[]
}

counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

# %%

mydfs = cut(mydfs, 'jet1_pt', 20, None)
# %%
mydfs = cut(mydfs, 'jet2_pt', 20, None)
# %%
mydfs = cut(mydfs, 'jet1_eta', -2.5, 2.5)
# %%
mydfs = cut(mydfs, 'jet2_eta', -2.5, 2.5)
# %%
counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)
# %%
for idx, df in enumerate(mydfs):
    m = (df.muon_pt>=9) & (abs(df.muon_eta)<=1.5) & (abs(df.muon_dxySig)>=6)
    mydfs[idx] = df[m]
counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)
# %%

mydfs = cut(mydfs, 'jet1_btagDeepFlavB', 0.2783, None)
mydfs = cut(mydfs, 'jet2_btagDeepFlavB', 0.2783, None)
counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)
# %%
mydfs = cut(mydfs, 'dijet_mass', 40, 300)
counts['Data'].append(len(mydfs[0]))
counts['ggH'].append((mydfs[1].sf*mydfs[1].PU_SF*mydfs[1].jet1_btag_central*mydfs[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs[2].sf*mydfs[2].PU_SF*mydfs[2].jet1_btag_central*mydfs[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)


print(counts)


# %%

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


print(counts)

# %%
categoriesLabels = ["Dijet pT < 100", "100 < Dijet pT < 160 & btag > T", "Dijet pT > 160 & btag > T", "Dijet pT > 160 (noTight)", "100 < Dijet pT < 160 (noTight)"]
categoriesLabels = ["0", "1", "2", "3", "4"]

ggH_fractions = np.array(counts['ggH'][-len(categoriesLabels):])   # / sum(counts['ggH'][-len(categoriesLabels):]) * 100
vbf_fractions = np.array(counts['VBF'][-len(categoriesLabels):])   # / sum(counts['VBF'][-len(categoriesLabels):]) * 100
data_fractions = np.array(counts['Data'][-len(categoriesLabels):]) #/ sum(counts['Data'][-len(categoriesLabels):]) * 100

# Create figure and axes
fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Create bar chart for Data
bars = ax[0].bar(categoriesLabels, data_fractions,color='black')#[['blue', 'green', 'red', 'purple', 'orange'])
#ax[0].set_title('Data Distribution')
ax[0].set_ylabel('Efficiency')
ax[0].set_yscale('log')
ax[0].set_ylim(0.001,10000)
ax[0].set_ylim(ax[0].get_ylim()[0], ax[0].get_ylim()[1]*2)
ax[0].set_xticklabels(categoriesLabels, rotation=0, ha='right')
for bar, value in zip(bars, data_fractions):
    ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_number_percentage(value), ha='center', va='bottom')



# Create bar chart for ggH
bars = ax[1].bar(categoriesLabels, ggH_fractions, color='red')#[['blue', 'green', 'red', 'purple', 'orange'])
#ax[1].set_title('ggH Distribution')
#ax[1].set_ylabel('Efficiency')
ax[1].set_yscale('log')
ax[1].set_ylim(0.001,10000)
ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1]*2)
ax[1].set_xticklabels(categoriesLabels, rotation=0, ha='right')
for bar, value in zip(bars, ggH_fractions):
    ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_number_percentage(value), ha='center', va='bottom')


# Create bar chart for VBF
bars = ax[2].bar(categoriesLabels, vbf_fractions, color='blue')#['blue', 'green', 'red', 'purple', 'orange'])
#ax[2].set_title('VBF Distribution')
#ax[2].set_ylabel('Efficiency')
ax[2].set_yscale('log')
ax[2].set_ylim(0.001,10000)
ax[2].set_ylim(ax[2].get_ylim()[0], ax[2].get_ylim()[1]*2)
ax[2].set_xticklabels(categoriesLabels, rotation=0, ha='right')
for bar, value in zip(bars, vbf_fractions):
    ax[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_number_percentage(value), ha='center', va='bottom')


plt.tight_layout()
fig.savefig("/t3home/gcelotto/ggHbb/stepsCheck/plots/CategoriesPt.png", bbox_inches='tight')


# %%
fig, ax = plt.subplots(figsize=(30, 6))
x = np.arange(len(preselectionSteps))

# Bar width
width = 0.25
bars_data = ax.bar(x[:len(preselectionSteps)] - width, counts['Data'][:len(preselectionSteps)], width, label='Data', color='black')
bars_ggH = ax.bar(x[:len(preselectionSteps)], counts['ggH'][:len(preselectionSteps)], width, label='ggH', color='red')
bars_vbf = ax.bar(x[:len(preselectionSteps)] + width, counts['VBF'][:len(preselectionSteps)], width, label='VBF', color='blue')




# Add formatted integer values on top of the bars
for idx, bars in enumerate([bars_data, bars_ggH, bars_vbf]):
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
fig.savefig("/t3home/gcelotto/ggHbb/stepsCheck/plots/preselectionSteps.png", bbox_inches='tight')

# %%

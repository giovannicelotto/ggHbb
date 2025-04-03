# %%
from functions import cut, cut_advanced, loadMultiParquet_Data_new, loadMultiParquet_v2, getCommonFilters, getDfProcesses_v2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
nn= True
import mplhep as hep
hep.style.use("CMS")
import torch, sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.scaleUnscale import scale, unscale
from functions import getDfProcesses_v2
def format_number_percentage(value):
    if value >99:
        return "%.1f"%value
    elif value >10:
        return "%.2f"%value
    elif 0.1<value <10:
        return "%.2f"%value
    else:
        return "%.3f"%value
    
# %%
isHiggsList = [43, 36]
dfProcessesMC = getDfProcesses_v2()[0]
xsections = dfProcessesMC.iloc[isHiggsList].xsection

featuresForTraining = list(np.load("/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/Mar21_2_0p0/featuresForTraining.npy"))
# %%
dfsData_, lumi, fileNumberListData = loadMultiParquet_Data_new(dataTaking=[1, 2], nReals=[10, 50], columns=['dijet_mass'], filters=None, returnFileNumberList=True)
dfsMC_, sumw, fileNumberListMC = loadMultiParquet_v2(paths=isHiggsList, nMCs=-1, columns=['dijet_mass',  'sf', 'PU_SF','genWeight', 'jet1_btag_central'], returnNumEventsTotal=True, filters=None, returnFileNumberList=True)
# %%
dfsData_f, lumi = loadMultiParquet_Data_new(dataTaking=[1, 2], nReals=[10, 50], columns=featuresForTraining+['jet1_btagDeepFlavB','jet2_btagDeepFlavB','muon_eta', 'dijet_mass'], filters=getCommonFilters(btagTight=False),selectFileNumberList=fileNumberListData )
dfsMC_f, sumw_f = loadMultiParquet_v2(paths=isHiggsList, nMCs=-1, columns=featuresForTraining+[ 'sf', 'PU_SF','jet1_btag_central','jet2_btagDeepFlavB', 'genWeight', 'jet1_btagDeepFlavB','muon_eta', 'dijet_mass'], returnNumEventsTotal=True, filters=getCommonFilters(btagTight=False), selectFileNumberList=fileNumberListMC)
# %%
# Start counting the unfiltered for first element
data_f=pd.concat(dfsData_f)
dfsMC_[0].jet1_btag_central = np.where(dfsMC_[0].jet1_btag_central.values<-9000, 1, dfsMC_[0].jet1_btag_central.values)
dfsMC_[1].jet1_btag_central = np.where(dfsMC_[1].jet1_btag_central.values<-9000, 1, dfsMC_[1].jet1_btag_central.values)
mydfs = [data_f,dfsMC_f[0], dfsMC_f[1] ]
mydfs_unfiltered = [pd.concat(dfsData_),dfsMC_[0], dfsMC_[1] ]
counts={
    'Data':[],
    'ggH':[],
    'VBF':[]}

counts['Data'].append(len(mydfs_unfiltered[0]))
counts['ggH'].append((mydfs_unfiltered[1].sf*mydfs_unfiltered[1].PU_SF*mydfs_unfiltered[1].jet1_btag_central*mydfs_unfiltered[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_unfiltered[2].sf*mydfs_unfiltered[2].PU_SF*mydfs_unfiltered[2].jet1_btag_central*mydfs_unfiltered[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

mydfs = cut(mydfs, 'dijet_mass', 40, 300)
# %%
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


if nn:
    
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
        (df['dijet_pt'] >= 100) & (df['dijet_pt'] < 160),  # Condition for 1
        (df['dijet_pt'] >= 160)   # Condition for 2
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
    #
    modelName = "Mar21_1_0p0"
    #featuresForTraining.remove(['dijet_mass'])
    modelDir="/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/%s"%modelName
    mydfs_1[0]  = scale(mydfs[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" ,fit=False)
    mydfs_1[1]  = scale(mydfs[1], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" ,fit=False)
    mydfs_1[2]  = scale(mydfs[2], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" ,fit=False)
#
    data_tensor = torch.tensor(np.float32(mydfs_1[0][featuresForTraining].values)).float()
    mydfs_1[0]['NN1'] = model1(data_tensor).detach().numpy()
    data_tensor = torch.tensor(np.float32(mydfs_1[1][featuresForTraining].values)).float()
    mydfs_1[1]['NN1'] = model1(data_tensor).detach().numpy()
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
#
    data_tensor = torch.tensor(np.float32(mydfs_2[1][featuresForTraining].values)).float()
    mydfs_2[1]['NN2'] = model2(data_tensor).detach().numpy()
    data_tensor = torch.tensor(np.float32(mydfs_2[0][featuresForTraining].values)).float()
    mydfs_2[0]['NN2'] = model2(data_tensor).detach().numpy()
    data_tensor = torch.tensor(np.float32(mydfs_2[2][featuresForTraining].values)).float()
    mydfs_2[2]['NN2'] = model2(data_tensor).detach().numpy()
#
    mydfs_2[0]  = unscale(mydfs_2[0], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" )
    mydfs_2[1]  = unscale(mydfs_2[1], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" )
    mydfs_2[2]  = unscale(mydfs_2[2], featuresForTraining=featuresForTraining, scalerName= modelDir + "/model/myScaler.pkl" )

# %%
mydfs_1 = cut(mydfs_1, 'dijet_pt', 100-1e-12, 160)
mydfs_1 = cut(mydfs_1, 'jet1_btagDeepFlavB', 0.71, None)
mydfs_1 = cut(mydfs_1, 'jet2_btagDeepFlavB', 0.71, None)
mydfs_2 = cut(mydfs_2, 'dijet_pt', 160-1e-6, None)
mydfs_2 = cut(mydfs_2, 'jet1_btagDeepFlavB', 0.71, None)
mydfs_2 = cut(mydfs_2, 'jet2_btagDeepFlavB', 0.71, None)

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
counts["Data"]=list(counts["Data"])
counts["ggH"]=list(counts["ggH"])
counts["VBF"]=list(counts["VBF"])

counts['Data'].append(len(mydfs_1[0]))
counts['ggH'].append((mydfs_1[1].sf*mydfs_1[1].PU_SF*mydfs_1[1].jet1_btag_central*mydfs_1[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_1[2].sf*mydfs_1[2].PU_SF*mydfs_1[2].jet1_btag_central*mydfs_1[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

mydfs_1 = cut(mydfs_1, 'dijet_pt', 100-1e-12, 160)

counts['Data'].append(len(mydfs_1[0]))
counts['ggH'].append((mydfs_1[1].sf*mydfs_1[1].PU_SF*mydfs_1[1].jet1_btag_central*mydfs_1[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_1[2].sf*mydfs_1[2].PU_SF*mydfs_1[2].jet1_btag_central*mydfs_1[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)

mydfs_1 = cut(mydfs_1, 'NN1', 0.575,None)


counts['Data'].append(len(mydfs_1[0]))
counts['ggH'].append((mydfs_1[1].sf*mydfs_1[1].PU_SF*mydfs_1[1].jet1_btag_central*mydfs_1[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_1[2].sf*mydfs_1[2].PU_SF*mydfs_1[2].jet1_btag_central*mydfs_1[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)


mydfs_2p = cut(mydfs_2, 'dijet_pt', 160-1e-6, None)
mydfs_2p = cut(mydfs_2p, 'NN2', 0.327, 0.75)
counts['Data'].append(len(mydfs_2p[0]))
counts['ggH'].append((mydfs_2p[1].sf*mydfs_2p[1].PU_SF*mydfs_2p[1].jet1_btag_central*mydfs_2p[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_2p[2].sf*mydfs_2p[2].PU_SF*mydfs_2p[2].jet1_btag_central*mydfs_2p[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)


mydfs_3 = cut(mydfs_2, 'dijet_pt', 160-1e-6, None)
mydfs_3 = cut(mydfs_3, 'NN2', 0.75,None)
counts['Data'].append(len(mydfs_3[0]))
counts['ggH'].append((mydfs_3[1].sf*mydfs_3[1].PU_SF*mydfs_3[1].jet1_btag_central*mydfs_3[1].genWeight).sum()/sumw[0]*xsections.iloc[0]*1000)
counts['VBF'].append((mydfs_3[2].sf*mydfs_3[2].PU_SF*mydfs_3[2].jet1_btag_central*mydfs_3[2].genWeight).sum()/sumw[1]*xsections.iloc[1]*1000)


# %%
for idx, process in enumerate(counts.keys()):
    counts[process]=np.array(counts[process])/(counts[process][0])*100



# %%
categories=['Cat1', 'Cat2', 'Cat3']
x = np.arange(len(categories))
# %%
fig, ax = plt.subplots(figsize=(30, 6))
# Bar width
width = 0.25
bars_data = ax.bar(x - width, counts['Data'][-3:], width, label='Data', color='black')
bars_ggH = ax.bar(x, counts['ggH'][-3:], width, label='ggH', color='red')
bars_vbf = ax.bar(x + width, counts['VBF'][-3:], width, label='VBF', color='blue')


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
        return "%.5f"%value
    elif 0.001<value <0.1:
        return "%.5f"%value
    else:
        return "%.5f"%value

# Add formatted integer values on top of the bars
for bars in [bars_data, bars_ggH, bars_vbf]:
    for bar in bars:
        height = bar.get_height()
        formatted_value = format_number_percentage(height)
        ax.text(bar.get_x() + bar.get_width() / 2, height, formatted_value, ha='center', va='bottom', fontsize=30)


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

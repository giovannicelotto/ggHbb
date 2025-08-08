# %%
import glob
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from functions import *


dfProcesses = getDfProcesses_v2()[0]
MCList = [35,19,20,21,22]

commonNano = '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ZJetsToQQ_noTrig2025Jun03'
dict_processes = {
    'nanoPaths' : [
        commonNano+'/ZJetsToQQ_HT-100to200',
                   commonNano+'/ZJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8',
                   commonNano+'/ZJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8',
                   commonNano+'/ZJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
                   commonNano+'/ZJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8'
                   ],
    'xsections': dfProcesses.xsection[MCList].values,
    'process' : dfProcesses.process[MCList].values

}
#assert (dict_processes['xsections'][0]>5e3) & ('100to200' in dict_processes['nanoPaths'][0])

# %%
pt_dict, weights_dict, LHE_HT_dict, nPartons_dict, total_genEventSumw_dict, LHE_ET_dict = {}, {}, {}, {}, {}, {}

for folder, xsection, process in zip(dict_processes['nanoPaths'], dict_processes['xsections'], dict_processes['process']):
    nFiles = 150 if (xsection>5e3) else 30
    fileNames = glob.glob(folder + "/**/*.root", recursive=True)[:nFiles]
    print(fileNames)
    print(process, f" Files : {len(fileNames)}")


    
    pt_list = []
    weight_list = []
    nPartons_list = []
    LHE_HT_list = []
    LHE_ET_list = []


    total_genEventSumw = 0

    for fname in fileNames:
        print(fileNames.index(fname)+1, "/", len(fileNames))
        with uproot.open(fname) as file:
            # Compute normalization weight from Runs tree
            if "Runs" in file:
                genEventSumw = file["Runs"].arrays("genEventSumw", library="np")
                file_genEventSumw = np.sum(genEventSumw["genEventSumw"])
            else:
                print(f"Runs tree not found in file: {fname}")
                file_genEventSumw = 0

            total_genEventSumw += file_genEventSumw

            # Read needed branches
            tree = file["Events"]
            branches = tree.arrays(
                ["genWeight",
                 "LHEPart_status", "LHEPart_pdgId", "LHEPart_pt", "LHE_HT", "LHE_HTIncoming"],
                library="ak"
            )

            # Z boson selection
            nOutgoingPartons = ak.sum((branches["LHEPart_status"]==1) & (((abs(branches["LHEPart_pdgId"])>=1) & (abs(branches["LHEPart_pdgId"])<=5)) | (abs(branches["LHEPart_pdgId"])==21)), axis=1)
            maskZ = (branches["LHEPart_status"] == 2) & (branches["LHEPart_pdgId"] == 23)
            maskZ_Event = ak.any((maskZ), axis=1) 

            pt_Z = ak.flatten(branches["LHEPart_pt"][maskZ])
            weight_Z = branches["genWeight"][maskZ_Event]*xsection  # flat weight, one per Z

            pt_list.append(pt_Z)
            weight_list.append(weight_Z)
            nPartons_list.append(nOutgoingPartons[maskZ_Event])
            LHE_HT_list.append(branches["LHE_HT"][maskZ_Event])
            LHE_ET_list.append(ak.sum(branches["LHEPart_pt"][branches["LHEPart_status"]==1], axis=1)[maskZ_Event])

    pt_dict[process] = pt_list
    weights_dict[process] = weight_list
    nPartons_dict[process] = nPartons_list
    LHE_HT_dict[process] = LHE_HT_list
    LHE_ET_dict[process] = LHE_ET_list
    total_genEventSumw_dict[process] = total_genEventSumw
# %%
for process in pt_dict.keys():
    pt_dict[process] = ak.concatenate(pt_dict[process])
    weights_dict[process] = ak.concatenate(weights_dict[process])/total_genEventSumw_dict[process] * 1/(0.01*(11.6*2 + 15.6*3))
    nPartons_dict[process] = ak.concatenate(nPartons_dict[process])
    LHE_HT_dict[process] = ak.concatenate(LHE_HT_dict[process])
    LHE_ET_dict[process] = ak.concatenate(LHE_ET_dict[process])

# %%
from hist import Hist
import yaml

with open('/t3home/gcelotto/ggHbb/Z_kfactor/config.yml', 'r') as file:
    config = yaml.safe_load(file)

min_LHE_HT = config['min_LHE_HT']
min_LHE_ET = config['min_LHE_ET']
min_Z_pt = config['min_Z_PT']
bins = config['bins']
bin_widths = np.diff(bins)

# Create histogram
hist_pt = Hist.new.Variable(bins, name="pt", label="LHE Z pT [GeV]").Weight()
hist_LHE_HT = Hist.new.Reg(35, 0, 800, name="LHE_HT", label="LHE HT").Weight()
hist_LHE_ET = Hist.new.Reg(35, 0, 800, name="LHE_ET", label="LHE ET").Weight()
hist_LHE_ET_addPartons = Hist.new.Reg(35, 0, 800, name="LHE_ET_addPartons", label="LHE_ET - Z pT").Weight()
h_Partons = Hist.new.Reg(7, 0, 7, name="nOutgoingPartons", label="LHE n Outgoing Partons").Weight()




# %%
total_weights = 0
pt_all = np.array([])
nPartons_all = np.array([])
LHE_HT_all = np.array([])
LHE_ET_all = np.array([])
LHE_ET_addPartons_all = np.array([])
weights_all = np.array([])
# Loop over datasets
for process in list(pt_dict.keys())[:]:
    pt = ak.to_numpy(pt_dict[process])
    weights = ak.to_numpy(weights_dict[process])
    nPartons = ak.to_numpy(nPartons_dict[process])
    LHE_HT = ak.to_numpy(LHE_HT_dict[process])
    LHE_ET = ak.to_numpy(LHE_ET_dict[process])
    LHE_ET_addPartons  = LHE_ET - pt

    furtherMask = (LHE_HT>min_LHE_HT) & (LHE_ET>min_LHE_ET) & (pt>min_Z_pt) #& (nPartons<=5)
    
    weights = weights[furtherMask]
    pt = pt[furtherMask]
    print("max pt", ak.max(pt))
    LHE_HT = LHE_HT[furtherMask]
    LHE_ET = LHE_ET[furtherMask]
    nPartons = nPartons[furtherMask]
    LHE_ET_addPartons=LHE_ET_addPartons[furtherMask]


    pt_all = np.concatenate([pt_all, pt])
    
    nPartons_all = np.concatenate([nPartons_all, nPartons])
    LHE_HT_all = np.concatenate([LHE_HT_all, LHE_HT])
    LHE_ET_all = np.concatenate([LHE_ET_all, LHE_ET])
    LHE_ET_addPartons_all = np.concatenate([LHE_ET_addPartons_all, LHE_ET_addPartons])
    weights_all = np.concatenate([weights_all, weights])

    total_weights +=np.sum(weights)
# %%
    # Fill histogram with scaled weights
hist_pt.fill(pt=np.clip(pt_all, bins[0], bins[-1]), weight=weights_all)
hist_LHE_HT.fill(LHE_HT=np.clip(LHE_HT_all, hist_LHE_HT.axes[0].edges[0], hist_LHE_HT.axes[0].edges[-1]-1), weight=weights_all)
hist_LHE_ET.fill(LHE_ET=np.clip(LHE_ET_all, hist_LHE_ET.axes[0].edges[0], hist_LHE_ET.axes[0].edges[-1]-1), weight=weights_all)
hist_LHE_ET_addPartons.fill(LHE_ET_addPartons=np.clip(LHE_ET_addPartons_all, hist_LHE_ET_addPartons.axes[0].edges[0], hist_LHE_ET_addPartons.axes[0].edges[-1]), weight=weights_all)

h_Partons.fill(nOutgoingPartons=np.clip(nPartons_all, 0, 7), weight=weights_all)

# %%
# Normalize to bin width
counts = hist_pt.values() / bin_widths
errors = np.sqrt(hist_pt.variances()) / bin_widths
bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2

# Plot
fig, ax = plt.subplots(1, 5, figsize=(19, 5))
bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
ax[0].stairs(values=counts, edges=bins, label="Z to QQ", linewidth=1.5)
ax[0].errorbar(bin_centers, counts, yerr=errors, fmt='.', color='black', capsize=2)
ax[0].set_xlabel("LHE Z pT [GeV]")
ax[0].set_ylabel("Entries / GeV")
ax[0].text(x=0.9, y=0.9, s=f"Entries : {len(pt)}\n Weighted Entries : {total_weights:.0f}", transform=ax[0].transAxes, ha='right')
ax[0].legend()
ax[0].set_yscale('log')
ax[0].text(x=0.9, y=0.8, s=f"Min Z pT = {ak.min(pt_all):.0f} GeV", transform=ax[0].transAxes, ha='right')


h_Partons.plot1d(ax=ax[1])
ax[1].set_ylim(-1000, ax[1].get_ylim()[1])
ax[1].text(x=0.9, y=0.9, s=f"Entries : {len(pt)}\n Weighted Entries : {total_weights:.0f}", transform=ax[1].transAxes, ha='right')




ax[2].text(x=0.9, y=0.9, s=f"Entries : {len(pt)}\n Weighted Entries : {total_weights:.0f}", transform=ax[2].transAxes, ha='right')
hist_LHE_HT.plot1d(ax=ax[2])
ax[2].text(x=0.9, y=0.8, s=f"Min LHE HT = {ak.min(LHE_HT_all):.0f} GeV", transform=ax[2].transAxes, ha='right')

ax[3].text(x=0.9, y=0.9, s=f"Entries : {len(pt)}\n Weighted Entries : {total_weights:.0f}", transform=ax[3].transAxes, ha='right')
hist_LHE_ET.plot1d(ax=ax[3])
ax[3].text(x=0.9, y=0.8, s=f"Min LHE ET = {ak.min(LHE_ET_all):.0f} GeV", transform=ax[3].transAxes, ha='right')

ax[4].text(x=0.9, y=0.9, s=f"Entries : {len(pt)}\n Weighted Entries : {total_weights:.0f}", transform=ax[4].transAxes, ha='right')
hist_LHE_ET_addPartons.plot1d(ax=ax[4])
ax[4].text(x=0.9, y=0.8, s=f"Min = {ak.min(LHE_ET_addPartons_all):.0f} GeV", transform=ax[4].transAxes, ha='right')


fig.savefig("/t3home/gcelotto/ggHbb/Z_kfactor/output/ZtoQQ.png", bbox_inches='tight')





# %%
# Save output if needed
counts_qq = {
    'ET_bins': hist_LHE_ET.axes[0].edges.tolist(),
    'ET_counts': hist_LHE_ET.values().tolist(),
    'ET_errors': np.sqrt(hist_LHE_ET.variances()).tolist(),
    'pT_bins': hist_pt.axes[0].edges.tolist(),
    'pT_counts': hist_pt.values().tolist(),
    'pT_errors': np.sqrt(hist_pt.variances()).tolist(),
}

import json
with open("/t3home/gcelotto/ggHbb/Z_kfactor/output/qq.json", "w") as f:
    json.dump(counts_qq, f, indent=4)  # indent=4 makes it human-readable

# %%

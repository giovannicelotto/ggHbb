# %%
import glob
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import glob
import yaml
import os
from functions import getDfProcesses_v2
# %%
processes = getDfProcesses_v2()[0].process.values
# %%
dict_for_plot = {
    "central_eta_L": {},
    "central_eta_M": {},
    "central_eta_T": {},
    "forward_eta_L": {},
    "forward_eta_M": {},
    "forward_eta_T": {},
                 }
maps_location = "/t3home/gcelotto/ggHbb/flatter/efficiency_btag_map/json_maps"
for process in processes:
    if os.path.exists(maps_location+f"/btag_efficiency_map_{process}_L.json"):
        map = yaml.load(open(maps_location+f"/btag_efficiency_map_{process}_L.json"), Loader=yaml.FullLoader)
        dict_for_plot["central_eta_L"][process]  = np.array(map['eff_map']['b'])[:,0]
        dict_for_plot["forward_eta_L"][process]  = np.array(map['eff_map']['b'])[:,1]
    if os.path.exists(maps_location+f"/btag_efficiency_map_{process}_M.json"):
        map = yaml.load(open(maps_location+f"/btag_efficiency_map_{process}_M.json"), Loader=yaml.FullLoader)
        dict_for_plot["central_eta_M"][process]  = np.array(map['eff_map']['b'])[:,0]
        dict_for_plot["forward_eta_M"][process]  = np.array(map['eff_map']['b'])[:,1]
    if os.path.exists(maps_location+f"/btag_efficiency_map_{process}_T.json"):
        map = yaml.load(open(maps_location+f"/btag_efficiency_map_{process}_T.json"), Loader=yaml.FullLoader)
        dict_for_plot["central_eta_T"][process]  = np.array(map['eff_map']['b'])[:,0]
        dict_for_plot["forward_eta_T"][process]  = np.array(map['eff_map']['b'])[:,1]
    #if "_T.json" in file:
    #    file.split("_T.json")
    #


    
#map = "/t3home/gcelotto/ggHbb/flatter/efficiency_btag_map/json_maps/btag_efficiency_map_DYToLL_L.json"
print(processes)
# %%
desired_processes = [
    "EWKZJets",
    'ZJetsToQQ_200to400',
    'ZJetsToQQ_400to600', 'ZJetsToQQ_600to800', 'ZJetsToQQ_800toInf',
    'ZJetsToQQ_100to200', 'VBFHToBB', 'GluGluHToBBMINLO'
]
#desired_processes = [
#    'GluGluH_M50_ToBB',
# 'GluGluH_M70_ToBB', 'GluGluH_M100_ToBB', 'GluGluH_M200_ToBB',
# 'GluGluH_M300_ToBB'
#]
from itertools import cycle


markers = cycle(['o', 's', '^', 'D', 'v'])

fig, ax = plt.subplots(1, 3, figsize=(15, 5),sharey=True)
labels = [f"{map['pt_bins'][i]}-{map['pt_bins'][i+1]}" for i in range(len(map['pt_bins']) - 1)]
for proc in dict_for_plot["central_eta_L"].keys():
    if proc not in desired_processes:
        continue
    mk = next(markers)
    mask = dict_for_plot["central_eta_L"][proc]!=0
    ax[0].plot((dict_for_plot["central_eta_L"][proc])[mask], label=proc,         linestyle='solid', marker=mk)


    mask = dict_for_plot["central_eta_M"][proc]!=0
    ax[1].plot((dict_for_plot["central_eta_M"][proc])[mask], label=proc,         linestyle='solid',marker=mk)


    mask = dict_for_plot["central_eta_T"][proc]!=0
    ax[2].plot((dict_for_plot["central_eta_T"][proc])[mask], label=proc,         linestyle='solid', marker=mk)

    ax[0].set_title("|$\eta$|<1.5, Loose WP", fontsize=12)
    ax[1].set_title("|$\eta$|<1.5, Medium WP", fontsize=12)
    ax[2].set_title("|$\eta$|<1.5, Tight WP", fontsize=12)

for ax_ in ax:
    ax_.set_xticks(range(len(labels)))
    ax_.set_xticklabels(labels, rotation=45)
    ax_.tick_params(axis='x', labelsize=10)
    ax_.set_ylim(0.5, 1.01)
    ax_.set_xlabel("Bin Edges [GeV]", fontsize=16)
ax[0].set_ylabel("Efficiency", fontsize=16)
ax[0].legend(fontsize=14)

# %%
markers = cycle(['o', 's', '^', 'D', 'v'])
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
labels = [f"{map['pt_bins'][i]}-{map['pt_bins'][i+1]}" for i in range(len(map['pt_bins']) - 1)]
for proc in dict_for_plot["forward_eta_L"].keys():
    #ls = next(linestyles)
    if proc not in desired_processes:
        continue
    mk = next(markers)

    mask = dict_for_plot["forward_eta_L"][proc]!=0
    ax[0].plot((dict_for_plot["forward_eta_L"][proc])[mask], label=proc,         linestyle='solid', marker=mk)


    mask = dict_for_plot["forward_eta_M"][proc]!=0
    ax[1].plot((dict_for_plot["forward_eta_M"][proc])[mask], label=proc,         linestyle='solid',marker=mk)



    mask = dict_for_plot["forward_eta_T"][proc]!=0
    ax[2].plot((dict_for_plot["forward_eta_T"][proc])[mask], label=proc,         linestyle='solid', marker=mk)

    ax[0].set_title("1.5<|$\eta$|<2.5, Loose WP", fontsize=12)
    ax[1].set_title("1.5<|$\eta$|<2.5, Medium WP", fontsize=12)
    ax[2].set_title("1.5<|$\eta$|<2.5, Tight WP", fontsize=12)

for ax_ in ax:
    ax_.set_xticks(range(len(labels)))
    ax_.set_xticklabels(labels, rotation=45)
    ax_.tick_params(axis='x', labelsize=10)
    ax_.set_ylim(0.2, 1.01)
    ax_.set_xlabel("Bin Edges [GeV]", fontsize=16)
ax[0].set_ylabel("Efficiency",fontsize=16)
ax[0].legend(fontsize=14)
# %%









desired_processes = [
   'ST_s-channel-hadronic', 'ST_s-channel-leptononic',
       'ST_t-channel-antitop', 'ST_t-channel-top', 'ST_tW-antitop',
       'ST_tW-top', 'TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic',
]
#desired_processes = [
#    'GluGluH_M50_ToBB',
# 'GluGluH_M70_ToBB', 'GluGluH_M100_ToBB', 'GluGluH_M200_ToBB',
# 'GluGluH_M300_ToBB'
#]
from itertools import cycle


markers = cycle(['o', 's', '^', 'D', 'v'])

fig, ax = plt.subplots(1, 3, figsize=(15, 5),sharey=True)
labels = [f"{map['pt_bins'][i]}-{map['pt_bins'][i+1]}" for i in range(len(map['pt_bins']) - 1)]
for proc in dict_for_plot["central_eta_L"].keys():
    if proc not in desired_processes:
        continue
    mk = next(markers)
    mask = dict_for_plot["central_eta_L"][proc]!=0
    ax[0].plot((dict_for_plot["central_eta_L"][proc])[mask], label=proc,         linestyle='solid', marker=mk)


    mask = dict_for_plot["central_eta_M"][proc]!=0
    ax[1].plot((dict_for_plot["central_eta_M"][proc])[mask], label=proc,         linestyle='solid',marker=mk)


    mask = dict_for_plot["central_eta_T"][proc]!=0
    ax[2].plot((dict_for_plot["central_eta_T"][proc])[mask], label=proc,         linestyle='solid', marker=mk)

    ax[0].set_title("|$\eta$|<1.5, Loose WP", fontsize=12)
    ax[1].set_title("|$\eta$|<1.5, Medium WP", fontsize=12)
    ax[2].set_title("|$\eta$|<1.5, Tight WP", fontsize=12)

for ax_ in ax:
    ax_.set_xticks(range(len(labels)))
    ax_.set_xticklabels(labels, rotation=45)
    ax_.tick_params(axis='x', labelsize=10)
    ax_.set_ylim(0.5, 1.01)
    ax_.set_xlabel("Bin Edges [GeV]", fontsize=16)
ax[0].set_ylabel("Efficiency", fontsize=16)
ax[0].legend(fontsize=14)

# %%
markers = cycle(['o', 's', '^', 'D', 'v'])
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
labels = [f"{map['pt_bins'][i]}-{map['pt_bins'][i+1]}" for i in range(len(map['pt_bins']) - 1)]
for proc in dict_for_plot["forward_eta_L"].keys():
    #ls = next(linestyles)
    if proc not in desired_processes:
        continue
    mk = next(markers)

    mask = dict_for_plot["forward_eta_L"][proc]!=0
    ax[0].plot((dict_for_plot["forward_eta_L"][proc])[mask], label=proc,         linestyle='solid', marker=mk)


    mask = dict_for_plot["forward_eta_M"][proc]!=0
    ax[1].plot((dict_for_plot["forward_eta_M"][proc])[mask], label=proc,         linestyle='solid',marker=mk)



    mask = dict_for_plot["forward_eta_T"][proc]!=0
    ax[2].plot((dict_for_plot["forward_eta_T"][proc])[mask], label=proc,         linestyle='solid', marker=mk)

    ax[0].set_title("1.5<|$\eta$|<2.5, Loose WP", fontsize=12)
    ax[1].set_title("1.5<|$\eta$|<2.5, Medium WP", fontsize=12)
    ax[2].set_title("1.5<|$\eta$|<2.5, Tight WP", fontsize=12)

for ax_ in ax:
    ax_.set_xticks(range(len(labels)))
    ax_.set_xticklabels(labels, rotation=45)
    ax_.tick_params(axis='x', labelsize=10)
    ax_.set_ylim(0.2, 1.01)
    ax_.set_xlabel("Bin Edges [GeV]", fontsize=16)
ax[0].set_ylabel("Efficiency",fontsize=16)
ax[0].legend(fontsize=14)
# %%

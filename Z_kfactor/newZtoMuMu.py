# %%
import glob
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/DYToLL_noTrig2025Jun03/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/crab_DYJetsToLL/250603_114533/0000"
fileNames = glob.glob(folder + "/*.root")[:50]

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
        nOutgoingPartons = ak.sum((branches["LHEPart_status"]==1) & (((abs(branches["LHEPart_pdgId"])>=1) & (abs(branches["LHEPart_pdgId"])<=5))| (abs(branches["LHEPart_pdgId"])==21)), axis=1)
        maskZ = (branches["LHEPart_status"] == 2) & (branches["LHEPart_pdgId"] == 23)
        maskZ_Event = ak.any((maskZ), axis=1) 

        pt_Z = ak.flatten(branches["LHEPart_pt"][maskZ])
        weight_Z = branches["genWeight"][maskZ_Event]  # flat weight, one per Z

        pt_list.append(pt_Z)
        weight_list.append(weight_Z)
        nPartons_list.append(nOutgoingPartons[maskZ_Event])
        LHE_HT_list.append(branches["LHE_HT"][maskZ_Event])
        LHE_ET_list.append(ak.sum(branches["LHEPart_pt"][branches["LHEPart_status"]==1], axis=1)[maskZ_Event])

# %%
from hist import Hist

# Combine all results
pt_all_ = ak.concatenate(pt_list)
weight_all = ak.flatten(weight_list)
nPartons_all_ = ak.flatten(nPartons_list)
LHE_HT_all_ = ak.flatten(LHE_HT_list)
LHE_ET_all_= ak.flatten(LHE_ET_list)
LHE_ET_addPartons_all_ = LHE_ET_all_ - pt_all_

# Scale to xsec, lumi
# 3.3658 is BR Z to ll (ee)
# 2025.74 is the xsection of DYtoEE sample frop mll > 50
xsec = 2025.74  / (0.01 * 3.3658)
final_weights_ = weight_all * xsec  / total_genEventSumw  # convert to proper units

# Plot
import yaml

with open('/t3home/gcelotto/ggHbb/Z_kfactor/config.yml', 'r') as file:
    config = yaml.safe_load(file)

min_LHE_HT = config['min_LHE_HT']
min_LHE_ET = config['min_LHE_ET']
min_Z_PT = config['min_Z_PT']
bins = config['bins']

h_LHE_HT = Hist.new.Reg(35, 0, 800, name="LHE_HT", label="LHE HT").Weight()
h_LHE_ET = Hist.new.Reg(35, 0, 800, name="LHE_ET", label="LHE ET").Weight()
hPartons = Hist.new.Reg(7, 0, 7, name="nOutgoingPartons", label="n LHE OutgoingPartons").Weight()
h_PT = Hist.new.Variable(bins, name="pt", label="LHE Z pT [GeV]").Weight()
hist_LHE_ET_addPartons = Hist.new.Reg(35, 0, 800, name="LHE_ET_addPartons", label="LHE_ET - Z pT").Weight()


bin_widths = np.diff(bins)

furtherMask = (LHE_ET_all_>min_LHE_ET) & (pt_all_>min_Z_PT)

final_weights = final_weights_[furtherMask]
nPartons_all = nPartons_all_[furtherMask]
pt_all = pt_all_[furtherMask]
LHE_HT_all = LHE_HT_all_[furtherMask]
LHE_ET_all = LHE_ET_all_[furtherMask]
LHE_ET_addPartons_all = LHE_ET_addPartons_all_[furtherMask]

# Create hist.Hist object with proper binning

h_PT.fill(pt=np.clip(pt_all, bins[0], bins[-1]), weight=final_weights)

# Normalize counts by bin width
counts = h_PT.values() / bin_widths
errors = h_PT.variances()**0.5 / bin_widths


# %%
# Plot
fig, ax = plt.subplots(1, 5, figsize=(19, 5))
# Step-style histogram (like histtype="step")
ax[0].stairs(values=counts, edges=bins, label="Z To LL", linewidth=1.5)
# Add error bars at bin centers
bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
ax[0].errorbar(bin_centers, counts, yerr=errors, fmt='.', color='black', capsize=2)
ax[0].set_xlabel("LHE Z pT [GeV]")
ax[0].set_ylabel("Entries / GeV")
ax[0].text(x=0.9, y=0.9, s=f"Entries : {len(pt_all)}\n Weighted Entries : {ak.sum(final_weights):.0f}", transform=ax[0].transAxes, ha='right')
ax[0].legend()
ax[0].set_yscale('log')
ax[0].text(x=0.9, y=0.8, s=f"Min Z pT = {ak.min(pt_all):.0f} GeV", transform=ax[0].transAxes, ha='right')


hPartons.fill(nOutgoingPartons=np.clip(nPartons_all, 0, 6), weight=final_weights)
hPartons.plot1d(ax=ax[1])
ax[1].set_ylim(-1000, ax[1].get_ylim()[1])
ax[1].text(x=0.9, y=0.9, s=f"Entries : {len(nPartons_all)}\n Weighted Entries : {ak.sum(final_weights):.0f}", transform=ax[1].transAxes, ha='right')
ax[1].text(x=0.9, y=0.8, s=f"Min NOutgoingParton = {ak.min(nPartons_all):.0f}", transform=ax[1].transAxes, ha='right')



h_LHE_HT.fill(LHE_HT=np.clip(LHE_HT_all, h_LHE_HT.axes[0].edges[0], h_LHE_HT.axes[0].edges[-1]), weight=final_weights)
ax[2].text(x=0.9, y=0.9, s=f"Entries : {len(LHE_HT_all)}\n Weighted Entries : {ak.sum(final_weights):.0f}", transform=ax[2].transAxes, ha='right')
ax[2].text(x=0.9, y=0.8, s=f"Min LHE HT = {ak.min(LHE_HT_all):.0f} GeV", transform=ax[2].transAxes, ha='right')
h_LHE_HT.plot1d(ax=ax[2])


h_LHE_ET.fill(LHE_ET=np.clip(LHE_ET_all, h_LHE_ET.axes[0].edges[0], h_LHE_ET.axes[0].edges[-1]), weight=final_weights)
ax[3].text(x=0.9, y=0.9, s=f"Entries : {len(LHE_ET_all)}\n Weighted Entries : {ak.sum(final_weights):.0f}", transform=ax[3].transAxes, ha='right')
ax[3].text(x=0.9, y=0.8, s=f"Min LHE ET = {ak.min(LHE_ET_all):.0f} GeV", transform=ax[3].transAxes, ha='right')
h_LHE_ET.plot1d(ax=ax[3])


hist_LHE_ET_addPartons.fill(LHE_ET_addPartons=np.clip(LHE_ET_addPartons_all, hist_LHE_ET_addPartons.axes[0].edges[0], hist_LHE_ET_addPartons.axes[0].edges[-1]), weight=final_weights)
ax[4].text(x=0.9, y=0.9, s=f"Entries : {len(LHE_ET_all)}\n Weighted Entries : {ak.sum(final_weights):.0f}", transform=ax[4].transAxes, ha='right')
ax[4].text(x=0.9, y=0.8, s=f"Min = {ak.min(LHE_ET_addPartons_all):.0f} GeV", transform=ax[4].transAxes, ha='right')
hist_LHE_ET_addPartons.plot1d(ax=ax[4])




fig.savefig("/t3home/gcelotto/ggHbb/Z_kfactor/output/dyee.png", bbox_inches='tight')








# %%
# Save output
# Save output if needed
counts_dy = {
    'ET_bins': h_LHE_ET.axes[0].edges.tolist(),
    'ET_counts': h_LHE_ET.values().tolist(),
    'ET_errors': np.sqrt(h_LHE_ET.variances()).tolist(),
    'pT_bins': h_PT.axes[0].edges.tolist(),
    'pT_counts': h_PT.values().tolist(),
    'pT_errors': np.sqrt(h_PT.variances()).tolist(),
}
import json
with open("/t3home/gcelotto/ggHbb/Z_kfactor/output/dyee.json", "w") as f:
    json.dump(counts_dy, f, indent=4)  # indent=4 makes it human-readable

# %%

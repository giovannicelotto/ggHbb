# %%
import numpy as np
import uproot
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
import glob
from functions import getDfProcesses_v2
import yaml
# %%
openingFile = open("/t3home/gcelotto/ggHbb/documentation/plotScripts/nanoToFlat_cfg.yaml", "r")
cfg = yaml.load(openingFile, Loader=yaml.FullLoader)

grouped_pNs=cfg['processes_grouped']
pNs = [pN for group in grouped_pNs.values() for pN in group]
# %%

dfProcesses = getDfProcesses_v2()[0]

# %%
steps = {}
group_steps = {}


# %%
for pN in pNs:
    process = dfProcesses.process.iloc[pN]
    xsection = dfProcesses.xsection.iloc[pN]
    print(f"\n=== Processing: {process} ===")

    nanoPaths = dfProcesses.nanoPath.iloc[pN]
    flatPaths = dfProcesses.flatPath.iloc[pN]
    nanoFiles = glob.glob(nanoPaths + "/**/*.root", recursive=True)
    flatFiles = glob.glob(flatPaths + "/**/*.parquet", recursive=True)
    flat = pd.read_parquet(flatFiles)
    totalGW = 0.
    totalSW = 0.
    for idx, nanoFile in enumerate(nanoFiles):
        file = uproot.open(nanoFile)
        Events = file["Events"]
        Runs = file["Runs"]
        branches = Events.arrays("genWeight")
        genWeight = branches["genWeight"].to_numpy()
        totalSW += Runs['genEventSumw'].array()[0]
        totalGW += np.sum(genWeight)
        if idx % 10 == 0:
            print(idx+1, "/", len(nanoFiles), "files processed.")




    mask_kin_cuts = (flat.dijet_pt>=100) & (flat.muon_pt>=9) & (abs(flat.muon_dxySig)>=6) & (flat.dijet_mass>=50) & (flat.dijet_mass<=300) & (abs(flat.jet1_eta)<=2.5) & (abs(flat.jet2_eta)<=2.5) & (abs(flat.muon_eta)<=1.5)

    mask_notMM = ~((flat.jet1_btagDeepFlavB>0.2783) & (flat.jet2_btagDeepFlavB>0.2783))
    mask_notTT = ~((flat.jet1_btagDeepFlavB>0.71) & (flat.jet2_btagDeepFlavB>0.71)) & (flat.jet1_btagDeepFlavB>0.2783) & (flat.jet2_btagDeepFlavB>0.2783)
    mask_TT = (flat.jet1_btagDeepFlavB>0.71) & (flat.jet2_btagDeepFlavB>0.71)

    efficiency_wo_sf = (flat.flat_weight/flat.sf).sum()/(totalGW)
    efficiency_w_sf = (flat.flat_weight).sum()/(totalGW)
    efficiency_kin_cuts = (flat[mask_kin_cuts].flat_weight).sum()/(flat.flat_weight).sum()
    efficiency_TT = (flat[(mask_TT) & (mask_kin_cuts) ].flat_weight).sum()/(flat[mask_kin_cuts].flat_weight).sum()
    efficiency_notTT = (flat[(mask_notTT) & (mask_kin_cuts) ].flat_weight).sum()/(flat[mask_kin_cuts].flat_weight).sum()
    efficiency_notMM = (flat[(mask_notMM) & (mask_kin_cuts) ].flat_weight).sum()/(flat[mask_kin_cuts].flat_weight).sum()

    yield_mini = xsection * 41.6 * 1000
    yield_nano = totalGW * xsection / (totalSW) * 41.6 * 1000
    yield_flat = flat.flat_weight.sum() * xsection / (totalSW) * 41.6 * 1000
    yield_kin = flat.loc[mask_kin_cuts, "flat_weight"].sum() * xsection / totalSW * 41.6 * 1000
    yield_TT = flat.loc[mask_TT & mask_kin_cuts, "flat_weight"].sum() * xsection / totalSW  * 41.6 * 1000
    yield_notTT = flat.loc[mask_notTT & mask_kin_cuts, "flat_weight"].sum() * xsection / totalSW  * 41.6 * 1000
    yield_notMM = flat.loc[mask_notMM & mask_kin_cuts, "flat_weight"].sum() * xsection / totalSW  * 41.6 * 1000
    print("\nSummary:")
    print(f"  Nano → Flat efficiency (w/ SF)   : {efficiency_w_sf:.6e}")
    print(f"  Nano → Flat efficiency (w/o SF)  : {efficiency_wo_sf:.6e}")
    print(f"  Kinematic cuts efficiency       : {efficiency_kin_cuts:.6e}")
    print(f"  TT efficiency                   : {efficiency_TT:.6e}")
    print(f"  notTT efficiency                : {efficiency_notTT:.6e}")
    print(f"  notMM efficiency                : {efficiency_notMM:.6e}")
    print("Check ", (efficiency_TT + efficiency_notTT + efficiency_notMM))

    print("")
    print("  Yields:")
    print(f"    Mini               : {yield_mini:10.4e}")
    print(f"    Nano               : {yield_nano:10.4e}")
    print(f"    Kin. cuts (w/ SF)  : {yield_kin:10.4e}")
    print(f"    TT                 : {yield_TT:10.4e}")
    print(f"    notTT              : {yield_notTT:10.4e}")
    print(f"    notMM              : {yield_notMM:10.4e}")

    steps[process] = {
        "Total Yields BParking": yield_mini,
        "Trigger": yield_nano,
        "Kin. Cuts": yield_kin,
        "TT Region": yield_TT,
        "MM Region": yield_notTT,
        "notMM Region": yield_notMM,
    }
    for pr_group in list(grouped_pNs.keys()):
        if pN in grouped_pNs[pr_group]:
            if pr_group not in group_steps:
                group_steps[pr_group] = {
                    "Total Yields BParking": 0.0,
                    "Trigger": 0.0,
                    "Kin. Cuts": 0.0,
                    "TT Region": 0.0,
                    "MM Region": 0.0,
                    "notMM Region": 0.0,
                }
            group_steps[pr_group]["Total Yields BParking"] += yield_mini
            group_steps[pr_group]["Trigger"] += yield_nano
            group_steps[pr_group]["Kin. Cuts"] += yield_kin
            group_steps[pr_group]["TT Region"] += yield_TT
            group_steps[pr_group]["MM Region"] += yield_notTT
            group_steps[pr_group]["notMM Region"] += yield_notMM

# %%
np.save("/t3home/gcelotto/ggHbb/documentation/outputScripts_forPlotting/cutflow_mini_to_regions.npy", group_steps)

# %%

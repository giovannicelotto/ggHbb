# %%
import uproot
import numpy as np
import awkward as ak
# List of file paths
paths = [
"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_1.root",
"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_2.root",
"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_3.root",
"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_4.root",
"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_5.root",
"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_6.root",
"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_7.root",
"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_8.root",
"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_fiducial_JESsmearedNominal2025Apr09/ZZ_TuneCP5_13TeV-pythia8/crab_ZZ/250409_160546/0000/MC_fiducial_JESsmearedNominal_Run2_mc_2025Apr09_9.root"
    # ... add more as needed
]

# Concatenate the "Events" TTree from all files
df = uproot.concatenate([f"{path}:Events" for path in paths])
# %%
mask_genPart_Z = df["GenPart_pdgId"]==23
mask_genPart_Zdaughter = df["GenPart_pdgId"][df["GenPart_genPartIdxMother"]]==23
# %%
pdgId_daughters = df["GenPart_pdgId"][(mask_genPart_Zdaughter) & ~(mask_genPart_Z)]
nDecayMuMuBB=0
for daughterListEvent in pdgId_daughters:
    if (5 in daughterListEvent) & (-5 in daughterListEvent) & (13 in daughterListEvent) & (-13 in daughterListEvent):
        nDecayMuMuBB+=1
print(nDecayMuMuBB)
# %%
nDecayEEBB=0
for daughterListEvent in pdgId_daughters:
    if (5 in daughterListEvent) & (-5 in daughterListEvent) & (11 in daughterListEvent) & (-11 in daughterListEvent):
        nDecayEEBB+=1
print(nDecayEEBB)
# %%

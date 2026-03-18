# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import uproot
import awkward as ak
import glob
# %%
#folder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/AllMC2026Mar06/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8/crab_GluGluHToBBMINLO/260306_084143/0000"
folder="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data_50GeV2026Mar06/ParkingBPH1/crab_data_Run2018D_part1/260306_091733/0000"
fileNames = glob.glob(f"{folder}/*.root")
# %%
f = uproot.open(fileNames[0])
tree = f["Events"]
branches = tree.arrays()
# %%
Jet_pt = branches["Jet_pt"]
Jet_puId = branches["Jet_puId"]
Jet_jetId = branches["Jet_jetId"]
Jet_eta = branches["Jet_eta"]
Jet_phi = branches["Jet_phi"]
# %%
ak.sum(ak.sum((Jet_pt>20) & (Jet_jetId==6) & ((Jet_pt>50) | (Jet_puId>=4)) & (Jet_eta>-2.5) & (Jet_eta<-1.3) & (Jet_phi>-1.57) & (Jet_phi<-0.87), axis=1)>=1)/len(Jet_pt)
# %%

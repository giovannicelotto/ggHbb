# %%
import glob, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import getDfProcesses
import mplhep as hep
import uproot
hep.style.use("CMS")
import awkward as ak
# %%
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/EWKZJets2024Oct21/EWKZ2Jets_ZToQQ_TuneCP5_13TeV-madgraph-pythia8/crab_EWKZ2Jets_ZToQQ/241021_090229/0000"
nBB_tot = 0
numEvents_tot = 0
entries_tot =0
# %%
fileNames = glob.glob(path+"/*.root")
print(fileNames[0])
# %%
for fileName in fileNames:
    print(fileNames.index(fileName)+1, "/", len(fileNames))
    
    f = uproot.open(fileName)
    
    tree = f["Events"]
    branches = tree.arrays()
    
    lumiBlocks = f['LuminosityBlocks']
    numEvents = np.sum(lumiBlocks.arrays()['GenFilter_numEventsTotal'])
    numEvents_tot = numEvents_tot + numEvents
    entries_tot = entries_tot + tree.num_entries

    GenPart_pdgId = branches["GenPart_pdgId"]
    GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"]


    print("Entries : ", tree.num_entries)
    bQuarks = ak.sum(ak.sum((abs(GenPart_pdgId)==5) & ((GenPart_pdgId[GenPart_genPartIdxMother])==23), axis=1)==2).sum()
    cQuarks = ak.sum(ak.sum((abs(GenPart_pdgId)==4) & ((GenPart_pdgId[GenPart_genPartIdxMother])==23), axis=1)==2).sum()
    sQuarks = ak.sum(ak.sum((abs(GenPart_pdgId)==3) & ((GenPart_pdgId[GenPart_genPartIdxMother])==23), axis=1)==2).sum()
    uQuarks = ak.sum(ak.sum((abs(GenPart_pdgId)==2) & ((GenPart_pdgId[GenPart_genPartIdxMother])==23), axis=1)==2).sum()
    dQuarks = ak.sum(ak.sum((abs(GenPart_pdgId)==1) & ((GenPart_pdgId[GenPart_genPartIdxMother])==23), axis=1)==2).sum()
    #print("b quarks ", bQuarks)
    #print("c quarks ", cQuarks)
    #print("s quarks ", sQuarks)
    #print("u quarks ", uQuarks)
    #print("d quarks ", dQuarks)
    #print(bQuarks + cQuarks + sQuarks + uQuarks + dQuarks)
    #nBB = ak.sum(ak.sum((abs(GenPart_pdgId)==5) & ((GenPart_pdgId[GenPart_genPartIdxMother])==23), axis=1)==2)
    nBB_tot = nBB_tot + bQuarks


# %%
print(numEvents_tot)
print(entries_tot)
print(nBB_tot)
# %%
print(nBB_tot/entries_tot)
print(entries_tot/numEvents_tot)
# %%
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJetsBB/*.parquet")
df = pd.read_parquet(fileNames)
len(df)
# %%

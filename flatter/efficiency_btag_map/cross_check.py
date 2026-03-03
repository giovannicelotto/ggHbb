# %%
# This script does the same job as run_btagMaps.py but in an event-by-event way.
# It cross-checks how many events has at least one jet with a triggering muon with pt9 and ip6 and at least one more good jet.
#
#
#
#
import awkward  as ak
import uproot
# %%
fileName = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/MC_samples2026Jan15/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia8/crab_GluGluHToBBMINLO/260115_100126/0000/MC_samples_Run2_mc_2026Jan15_82.root"
f = uproot.open(fileName)
tree = f["Events"]
branches = tree.arrays(library="ak")
# %%
events = tree.num_entries
# %%
events_with_Trig_muon = 0
for ev in range(events):
    Jet_pt =  branches["Jet_pt"][ev]
    Jet_puId =  branches["Jet_puId"][ev]
    Jet_jetId =  branches["Jet_jetId"][ev]
    Jet_pt =  branches["Jet_pt"][ev]
    Muon_pt =  branches["Muon_pt"][ev]
    Muon_dxy =  branches["Muon_dxy"][ev]
    Muon_dxyErr =  branches["Muon_dxyErr"][ev]
    Muon_eta =  branches["Muon_eta"][ev]
    Jet_eta =  branches["Jet_eta"][ev]
    Jet_muonIdx1 =  branches["Jet_muonIdx1"][ev]
    Jet_muonIdx2 =  branches["Jet_muonIdx2"][ev]
    Muon_isTriggering =  branches["Muon_isTriggering"][ev]

    added = False
    for j in range(len(Jet_pt)):
        if (Jet_puId[j]>=4) | (Jet_pt[j]>50):
            if Jet_jetId[j]==6:
                if Jet_muonIdx1[j]>-1:
                    if ((Muon_isTriggering[Jet_muonIdx1[j]]) & (Muon_pt[Jet_muonIdx1[j]]>9) & (abs(Muon_eta[Jet_muonIdx1[j]])<1.5) & (abs(Muon_dxy[Jet_muonIdx1[j]] / Muon_dxyErr[Jet_muonIdx1[j]])>6)):
                        for j2 in range(len(Jet_pt)):
                            if j2==j:
                                continue
                            if (Jet_puId[j2]>=4) | (Jet_pt[j2]>50):
                                if Jet_jetId[j2]==6:
                                    events_with_Trig_muon+=1
                                    added=True
                                    break
                        if added==True:
                            break
                if added==False:
                    if Jet_muonIdx2[j]>-1:
                        if ((Muon_isTriggering[Jet_muonIdx2[j]]) & (Muon_pt[Jet_muonIdx2[j]]>9) & (abs(Muon_eta[Jet_muonIdx2[j]])<1.5) & (abs(Muon_dxy[Jet_muonIdx2[j]] / Muon_dxyErr[Jet_muonIdx2[j]])>6)):
                            for j2 in range(len(Jet_pt)):
                                if j2==j:
                                    continue
                                if (Jet_puId[j2]>=4) | (Jet_pt[j2]>50):
                                    if Jet_jetId[j2]==6:
                                        events_with_Trig_muon+=1
                                        added=True
                                        break
                            if added==True:
                                break
            
            else:
                continue
        else:
            continue
        


# %%

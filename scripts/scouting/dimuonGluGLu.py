# %%
import uproot
import glob
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append("/t3home/gcelotto/ggHbb/flatter/")
from treeFlatter import jetsSelector
# %%
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Mar05/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/crab_GluGluHToBB/240305_081723/0000/*.root")[:10]
mothers1 = []
mothers2 = []
totChargeL = []
for fN in fileNames:
    print("Opening %s"%fN)
    f = uproot.open(fN)
    tree = f['Events']
    branches = tree.arrays()
    maxEntries = tree.num_entries
    for ev in range(maxEntries):
        Muon_pt = branches["Muon_pt"][ev]
        Muon_isTriggering = branches["Muon_isTriggering"][ev]
        nMuon = branches["nMuon"][ev]
        Muon_charge = branches["Muon_charge"][ev]
        Muon_genPartIdx = branches["Muon_genPartIdx"][ev]
        GenPart_genPartIdxMother = branches["GenPart_genPartIdxMother"][ev]
        GenPart_pdgId            = branches["GenPart_pdgId"][ev]
        nJet                     = branches["nJet"][ev]
        Jet_eta                  = branches["Jet_eta"][ev]
        Jet_muonIdx1             = branches["Jet_muonIdx1"][ev]
        Jet_muonIdx2             = branches["Jet_muonIdx2"][ev]
        Jet_btagDeepFlavB        = branches["Jet_btagDeepFlavB"][ev]
        jetsToCheck = np.min((nJet, 4))

        totCharge = 0
        muonIdx1 = 999
        muonIdx2 = 999
        selected1, selected2 = 999, 999
        selected1, selected2, muonIdx1, muonIdx2 = jetsSelector(nJet, Jet_eta, Jet_muonIdx1,  Jet_muonIdx2, Muon_isTriggering, jetsToCheck, Jet_btagDeepFlavB)
        if np.sum(Muon_isTriggering)<=1:
            continue
        if muonIdx2==999:
            continue
        #    for mu in range(nMuon):
        #        if Muon_isTriggering[mu]:
        #            totCharge = totCharge + Muon_charge[mu]
        #            if muonIdx1==999:
        #                muonIdx1=mu
        #                continue
        #            elif muonIdx2==999:
        #                muonIdx2=mu
        #                if abs(totCharge)==2:
        #                    break
            #chcek the mother
        genPartMuon1 = Muon_genPartIdx[muonIdx1]
        genPartMuon2 = Muon_genPartIdx[muonIdx2]
        if abs(GenPart_pdgId[genPartMuon1])!=13:
            continue
        if abs(GenPart_pdgId[genPartMuon2])!=13:
            continue
        mothers1.append(GenPart_pdgId[GenPart_genPartIdxMother[genPartMuon1]])
        mothers2.append(GenPart_pdgId[GenPart_genPartIdxMother[genPartMuon2]])
        totChargeL.append(Muon_charge[muonIdx1] + Muon_charge[muonIdx2])
                        

            # loop over muons
            # check if triggering
            # check the sum of charge
            #if same sign
            # check the mother
# %%
def map_to_groups_letter(value):
    if abs(value) in [511, 521, 531, 541]:
        return 'B'
    elif abs(value) in [411, 421, 431]:
        return 'D'
    elif ((abs(value) > 3000) & (abs(value) < 4000)):
        return 'SB'
    elif ((abs(value) > 4000) & (abs(value) < 5000)):
        return 'CB'
    elif ((abs(value) > 5000) & (abs(value) < 6000)):
        return 'BB'
    elif (abs(value) == 15):
        return 'Tau'
    elif (abs(value) == 443):
        return 'JPsi'
    elif (abs(value) == 111):
        return 'Pi0'
    # Add more conditions as needed
    else:
        return 'Other'  # or any default value for unmatched cases
df = pd.DataFrame({
    'motherMuon1': mothers1,
    'motherMuon2': mothers2,
    'totCharge': totChargeL
})
# %%
df['mother1PDGClass'] = df['motherMuon1'].apply(map_to_groups_letter)
df['mother2PDGClass'] = df['motherMuon2'].apply(map_to_groups_letter)
df['HasD'] = df.apply(lambda row: 1 if row['mother1PDGClass'] == 'D' or row['mother2PDGClass'] == 'D' else 0, axis=1)
df['BothFromB'] = df.apply(lambda row: 1 if row['mother1PDGClass'] == 'B' and row['mother2PDGClass'] == 'B' else 0, axis=1)
df['NoneFromB'] = df.apply(lambda row: 1 if row['mother1PDGClass'] != 'B' and row['mother2PDGClass'] != 'B' else 0, axis=1)
df['BothFromD'] = df.apply(lambda row: 1 if row['mother1PDGClass'] == 'D' and row['mother2PDGClass'] == 'D' else 0, axis=1)
df['OneBOneD'] = df.apply(lambda row: 1 if (row['mother1PDGClass'] == 'B' and row['mother2PDGClass'] == 'D') or (row['mother1PDGClass'] == 'D' and row['mother2PDGClass'] == 'B') else 0, axis=1)
df['OneBOneNotB'] = df.apply(lambda row: 1 if (row['mother1PDGClass'] == 'B' and row['mother2PDGClass'] != 'B') or (row['mother1PDGClass'] != 'B' and row['mother2PDGClass'] == 'B') else 0, axis=1)
# %%
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 4, 4)

c = [df.NoneFromB.sum()/len(df), df.BothFromB.sum()/len(df), df.OneBOneNotB.sum()/len(df)]
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step')
ax.set_xticks(bins[:-1] + np.diff(bins) / 2)  # Center the ticks
ax.set_xticklabels(["None from B", "Both from B", "One From B One Not From B"], rotation=45)



for id, count in enumerate(c):
    if count==0:
        continue
    ax.text(x=bins[id]+0.25, y=0.4, s="%.1f %%"%(count*100))



# %%
fig, ax = plt.subplots(1, 1)
bins  = np.linspace(-0.5, 1.5, 3)
c = np.histogram(df.OneBOneD, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='Events with 2 TrigMuons')
ax.set_xlabel("One B One D")
ax.legend()
# %%

fig, ax = plt.subplots(1, 1)
bins  = np.linspace(-0.5, 1.5, 3)
c = np.histogram(df.BothFromB, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='All events with 2 Trig Muons')
ax.set_xlabel("Both From B")
ax.legend()
# %%

fig, ax = plt.subplots(1, 1)
bins  = np.linspace(-0.5, 1.5, 3)
c = np.histogram(df.BothFromD, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='All events with 2 Trig Muons')
ax.set_xlabel("Both From D")
ax.legend()
# %%
fig, ax = plt.subplots(1, 1)
bins  = np.linspace(-0.5, 1.5, 3)
c = np.histogram(df.OneBOneNotB, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='Events with 2 TrigMuons')
ax.set_xlabel("One B One Not B")
ax.legend()

# %%
fig, ax = plt.subplots(1, 1)
bins  = np.linspace(-2.5, 2.5, 6)
c = np.histogram(df[df.BothFromB==1].totCharge, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='Both Muons from B')
c = np.histogram(df[df.OneBOneNotB==1].totCharge, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='One from B One Not')
c = np.histogram(df[df.BothFromB==0].totCharge, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='None From B')
ax.set_xlabel("sum diMuon Charge")
ax.legend()





# %%
value_counts = df['mother1PDGClass'].value_counts()
value_counts=value_counts/np.sum(value_counts)
fig, ax = plt.subplots(1, 1)
value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
ax.set_title('mother Muon1 PDGClass')
ax.set_xlabel('PDG Classes')
ax.set_ylabel('Events')
plt.xticks(rotation=45)

#value_counts = df['mother1PDGClass'][abs(df.totCharge)==2].value_counts()
#value_counts.plot(kind='bar', color='red', edgecolor='black')
print("Mother 1 ", value_counts)
value_counts = df['mother2PDGClass'].value_counts()
value_counts=value_counts/np.sum(value_counts)
fig, ax = plt.subplots(1, 1)
value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
ax.set_title('mother Muon2 PDGClass')
ax.set_xlabel('PDG Classes')
ax.set_ylabel('Events')
plt.xticks(rotation=45)

#value_counts = df['mother1PDGClass'][abs(df.totCharge)==2].value_counts()
#value_counts.plot(kind='bar', color='red', edgecolor='black')
print("Mother 2 ", value_counts)
# %%

fig, ax = plt.subplots(1, 1)
bins  = np.linspace(-0.5, 1.5, 3)
c = np.histogram(df.HasD, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='All events with 2 Trig Muons')
c = np.histogram(df.HasD[abs(df.totCharge)==2], bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='|Sum| charge = 2')
ax.set_xlabel("At least one Muon from D")
ax.legend()


# %%
fig, ax = plt.subplots(1, 1)
bins  = np.linspace(-2.5, 2.5, 6)
c = np.histogram(df[df.BothFromB==1].totCharge, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='Events where both Muons from B')
ax.set_xlabel("totCharge")
ax.legend()
# %%
fig, ax = plt.subplots(1, 1)
bins  = np.linspace(-2.5, 2.5, 6)
c = np.histogram(df[df.HasD==1].totCharge, bins=bins)[0]
c=c/np.sum(c)
ax.hist(bins[:-1], bins=bins, weights=c, histtype=u'step', label='Events where at least one Muon from D')
ax.set_xlabel("totCharge")
ax.legend()
# %%

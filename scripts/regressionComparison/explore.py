# %%
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
# %%
# Get filenames
nFiles = -1
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched/GluGluHToBB/*.parquet")
fileNames = fileNames if nFiles==-1 else fileNames[:nFiles]
print("nFiles : %d"%len(fileNames))
# %%
# Read parquet files
df = pd.read_parquet(fileNames)

# %%

#
#
#               Jet 1
#
#



cuts = [15, 25, 40,
        60, 80, 100, 10**4]
nrow, ncol = 2, 3
fig, ax = plt.subplots(nrow, ncol, figsize=(20, 15))
binning = [20, 30, 30, 50, 60, 80]
for i in range(nrow):
    for j in range(ncol):
        bins = np.linspace(-binning[i*ncol+j], binning[i*ncol + j], 41)
        mask = (df.genJet1_pt > cuts[i*ncol + j]) & (df.genJet1_pt < cuts[i*ncol + j+1]) & (abs(df.jet1_eta)<2.5)
        df_ = df[mask]
        print(len(df_))
        creco = ax[i,j].hist(np.clip(df_.jet1_pt - df_.genJet1_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='reco')[0]
        cpnet = ax[i,j].hist(np.clip(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_PNetRegPtRawCorr - df_.genJet1_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='PNet')[0]
        cpart = ax[i,j].hist(np.clip(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_ParTAK4RegPtRawCorr - df_.genJet1_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='ParT')[0]
        ax[i, j].set_title(" %d<GenJet 1 pT <%d"%(cuts[i*ncol + j], cuts[i*ncol + j+1]))

        ax[i, j].errorbar(x=np.median(df_.jet1_pt - df_.genJet1_pt), y=np.max(creco), marker='o', color='C0')
        ax[i, j].errorbar(x=np.median(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_PNetRegPtRawCorr - df_.genJet1_pt), y=np.max(cpnet), marker='o', color='C1')
        ax[i, j].errorbar(x=np.median(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_ParTAK4RegPtRawCorr - df_.genJet1_pt), y=np.max(cpart), marker='o', color='C2')

ax[0, 0].legend()



# %%




bins = np.linspace(-70, 70, 101)

fig, ax = plt.subplots(1, 1)
mean_real = []
mean_raw = []
mean_reco = []

mean_pnet = []
mean_part = []
std_real = []
std_raw = []
std_reco = []

std_pnet = []
std_part = []
for i in range(nrow):
    for j in range(ncol):
        mask = (df.genJet1_pt > cuts[i*ncol + j]) & (df.genJet1_pt < cuts[i*ncol + j+1]) & (abs(df.jet1_eta)<2.5)
        df_ = df[mask]

        mean_real.append ( np.median(df_.genJet1_pt))

        mean_raw.append ( np.median(df_.jet1_pt* (1-df_.jet1_rawFactor) - df_.genJet1_pt))
        mean_reco.append ( np.median(df_.jet1_pt - df_.genJet1_pt))
        mean_pnet.append ( np.median(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_PNetRegPtRawCorr - df_.genJet1_pt))
        mean_part.append ( np.median(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_ParTAK4RegPtRawCorr - df_.genJet1_pt))

        std_real.append(np.std(df_.genJet1_pt))
        std_raw.append(np.std(df_.jet1_pt* (1-df_.jet1_rawFactor) - df_.genJet1_pt)/np.sqrt(len(df_)))
        std_reco.append(np.std(df_.jet1_pt - df_.genJet1_pt)/np.sqrt(len(df_)))
        std_pnet.append(np.std(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_PNetRegPtRawCorr - df_.genJet1_pt)/np.sqrt(len(df_)))
        std_part.append(np.std(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_ParTAK4RegPtRawCorr - df_.genJet1_pt)/np.sqrt(len(df_)))
cuts = np.array([15, 25, 40,
        60, 80, 100, 200])
mean_real=np.array(mean_real)
mean_raw=np.array(mean_raw)
mean_reco=np.array(mean_reco)
mean_pnet=np.array(mean_pnet)
mean_part=np.array(mean_part)

std_real=np.array(std_real)
std_raw=np.array(std_raw)
std_reco=np.array(std_reco)
std_pnet=np.array(std_pnet)
std_part=np.array(std_part)

#ax.errorbar((cuts[1:] + cuts[:-1])/2, 100*mean_raw/mean_real, xerr=(cuts[1:]-cuts[:-1])/2, yerr=100*std_raw/mean_real, linestyle='none', label='raw')
ax.errorbar((cuts[1:] + cuts[:-1])/2, 100*mean_reco/mean_real,xerr=(cuts[1:]-cuts[:-1])/2,  yerr=100*std_reco/mean_real, linestyle='none', label='reco')
ax.errorbar((cuts[1:] + cuts[:-1])/2, 100*mean_pnet/mean_real,xerr=(cuts[1:]-cuts[:-1])/2,  yerr=100*std_pnet/mean_real, linestyle='none', label='pnet')
ax.errorbar((cuts[1:] + cuts[:-1])/2, 100*mean_part/mean_real,xerr=(cuts[1:]-cuts[:-1])/2,  yerr=100*std_part/mean_real, linestyle='none', label='part')
ax.set_xlabel("Gen Jet1 pT")
ax.set_ylabel("(Reco pT - Gen Jet pT)/ GenJet pT [%]")
ax.set_ylim(-30, 30)
ax.legend()




# %%

#
#
#      Jet 2
#
#


cuts = [15, 25, 40,
        60, 80, 100, 10**4]
nrow, ncol = 2, 3
fig, ax = plt.subplots(nrow, ncol, figsize=(20, 15))

for i in range(nrow):
    for j in range(ncol):
        bins = np.linspace(-binning[i*ncol+j], binning[i*ncol + j], 41)
        mask = (df.genJet2_pt > cuts[i*ncol + j]) & (df.genJet2_pt < cuts[i*ncol + j+1]) & (abs(df.jet2_eta)<2.5)
        df_ = df[mask]

        creco = ax[i,j].hist(np.clip(df_.jet2_pt - df_.genJet2_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='reco')[0]
        cpnet = ax[i,j].hist(np.clip(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_PNetRegPtRawCorr - df_.genJet2_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='PNet Neutrinos')[0]
        cpart = ax[i,j].hist(np.clip(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr - df_.genJet2_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='ParT Neutrinos')[0]
        ax[i, j].set_title(" %d<GenJet 2 pT <%d"%(cuts[i*ncol + j], cuts[i*ncol + j+1]))

        ax[i, j].errorbar(x=np.median(df_.jet2_pt - df_.genJet2_pt), y=np.max(creco), marker='o', color='C0')
        ax[i, j].errorbar(x=np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_PNetRegPtRawCorr - df_.genJet2_pt), y=np.max(cpnet), marker='o', color='C1')
        ax[i, j].errorbar(x=np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr - df_.genJet2_pt), y=np.max(cpart), marker='o', color='C2')

ax[0, 0].legend()


# %%


bins = np.linspace(-70, 70, 101)


fig, ax = plt.subplots(1, 1)
mean_real = []
mean_raw = []
mean_reco = []
mean_pnet = []
mean_part = []
std_real = []
std_raw = []
std_reco = []
std_pnet = []
std_part = []
for i in range(nrow):
    for j in range(ncol):
        mask = (df.genJet2_pt > cuts[i*ncol + j]) & (df.genJet2_pt < cuts[i*ncol + j+1]) & (abs(df.jet2_eta)<2.5)
        df_ = df[mask]

        mean_real.append ( np.median(df_.genJet2_pt))

        mean_raw.append ( np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) - df_.genJet2_pt))
        mean_reco.append ( np.median(df_.jet2_pt - df_.genJet2_pt))
        mean_pnet.append ( np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_PNetRegPtRawCorr - df_.genJet2_pt))
        mean_part.append ( np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr - df_.genJet2_pt))

        std_real.append(np.std(df_.genJet2_pt))
        std_raw.append(np.std(df_.jet2_pt* (1-df_.jet2_rawFactor) - df_.genJet2_pt)/np.sqrt(len(df_)))
        std_reco.append(np.std(df_.jet2_pt - df_.genJet2_pt)/np.sqrt(len(df_)))
        std_pnet.append(np.std(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_PNetRegPtRawCorr - df_.genJet2_pt)/np.sqrt(len(df_)))
        std_part.append(np.std(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr - df_.genJet2_pt)/np.sqrt(len(df_)))

cuts = np.array([15, 25, 40,
        60, 80, 100, 200])
mean_real=np.array(mean_real)
mean_raw=np.array(mean_raw)
mean_reco=np.array(mean_reco)
mean_pnet=np.array(mean_pnet)
mean_part=np.array(mean_part)

std_real=np.array(std_real)
std_reco=np.array(std_reco)
std_raw=np.array(std_raw)
std_pnet=np.array(std_pnet)
std_part=np.array(std_part)

ax.errorbar((cuts[1:] + cuts[:-1])/2, 100*mean_reco/mean_real, xerr=(cuts[1:]-cuts[:-1])/2,  yerr=100*std_reco/mean_real, linestyle='none', label='reco')
ax.errorbar((cuts[1:] + cuts[:-1])/2, 100*mean_pnet/mean_real, xerr=(cuts[1:]-cuts[:-1])/2,  yerr=100*std_pnet/mean_real, linestyle='none', label='pnet')
ax.errorbar((cuts[1:] + cuts[:-1])/2, 100*mean_part/mean_real, xerr=(cuts[1:]-cuts[:-1])/2,  yerr=100*std_part/mean_real, linestyle='none', label='part')
ax.set_xlabel("Gen Jet2 pT")
ax.set_ylabel("(Reco pT - Gen Jet pT)/ GenJet pT [%]")
ax.set_ylim(-30, 30)
ax.legend()


# %%

mask = (abs(df.jet1_eta)<2.5) & (abs(df.jet2_eta)<2.5) & (df.genJet1_pt>0) & (df.genJet2_pt>0) 


fig, ax = plt.subplots(1, 1)
bins = np.linspace(60, 180, 41)
ax.set_title("|eta|<2.5 & pT>?")
ax.hist(np.clip(df.dijet_mass_reco[mask], bins[0], bins[-1]), bins=bins, histtype='step', label='reco')
ax.hist(np.clip(df.dijet_mass_pnet[mask], bins[0], bins[-1]), bins=bins, histtype='step', label='pnet')
ax.hist(np.clip(df.dijet_mass_parT[mask], bins[0], bins[-1]), bins=bins, histtype='step', label='parT')
ax.hist(np.clip(df.genDijet_mass[mask], bins[0], bins[-1]), bins=bins, histtype='step', label='GenDijet')
ax.legend()
# %%
# dijet mass in bin of pt

bins = np.linspace(50, 180, 41)
cuts = np.linspace(0, 200, 20)

fig, ax = plt.subplots(1, 1)
mean_real = []
mean_raw  = []
mean_reco = []

mean_pnet = []
mean_part = []
std_real  = []
std_raw   = []
std_reco  = []

std_pnet  = []
std_part  = []
for c in cuts:
        mask = (df.jet1_pt > c) & (df.jet2_pt > c) & (abs(df.jet2_eta)<2.5) & (abs(df.jet1_eta)<2.5)
        df_ = df[mask]

        mean_real.append ( np.median(df_.genDijet_mass))

        mean_reco.append ( np.median(df_.dijet_mass_reco))
        mean_pnet.append ( np.median(df_.dijet_mass_pnet))
        mean_part.append ( np.median(df_.dijet_mass_parT))

        std_real.append(np.std(df_.genDijet_mass)/np.sqrt(len(df_)))
        std_reco.append(np.std(df_.dijet_mass_reco)/np.sqrt(len(df_)))
        std_pnet.append(np.std(df_.dijet_mass_pnet)/np.sqrt(len(df_)))
        std_part.append(np.std(df_.dijet_mass_parT)/np.sqrt(len(df_)))



mean_real=np.array(mean_real)
mean_reco=np.array(mean_reco)
mean_pnet=np.array(mean_pnet)
mean_part=np.array(mean_part)

std_real=np.array(std_real)
std_reco=np.array(std_reco)
std_pnet=np.array(std_pnet)
std_part=np.array(std_part)

x = cuts


ax.plot(x, mean_reco, label='reco', color='blue')
ax.plot(x, mean_pnet, label='pnet', color='orange')
ax.plot(x, mean_part, label='part', color='green')
ax.plot(x, mean_real, label='gen', color='gray')
#
## Add the shaded error regions
ax.fill_between(x, mean_reco - std_reco, mean_reco + std_reco, color='blue', alpha=0.2)
ax.fill_between(x, mean_pnet - std_pnet, mean_pnet + std_pnet, color='orange', alpha=0.2)
ax.fill_between(x, mean_part - std_part, mean_part + std_part, color='green', alpha=0.2)
ax.fill_between(x, mean_real - std_real, mean_real + std_real, color='gray', alpha=0.2)
ax.hlines(y=125.09, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dotted', color='black')
ax.set_xlim(cuts[0], cuts[-1])
ax.set_xlabel("jet1&2 pt bin")
ax.set_ylabel("Dijet Mass Reco")

ax.legend()


# %%
# genjet only greater than without having lower that
fig, ax = plt.subplots(1, 1)
mean_real = []
mean_raw  = []
mean_reco = []
mean_pnet = []
mean_part = []
std_real  = []
std_raw   = []
std_reco  = []
std_pnet  = []
std_part  = []
cuts = np.linspace(0, 200, 20)
for c in cuts:
        print(c)
        mask = (df.jet1_pt>c) & (df.jet2_pt > c) & (abs(df.jet2_eta)<2.5) & (abs(df.jet1_eta)<2.5)
        df_ = df[mask]

        mean_real.append (np.median(df_.genDijet_mass))

        mean_reco.append ( np.median(df_.dijet_mass_reco))
        mean_pnet.append ( np.median(df_.dijet_mass_pnet))
        mean_part.append ( np.median(df_.dijet_mass_parT))

        std_reco.append(np.std(df_.dijet_mass_reco)/np.sqrt(len(df_)))
        std_pnet.append(np.std(df_.dijet_mass_pnet)/np.sqrt(len(df_)))
        std_part.append(np.std(df_.dijet_mass_parT)/np.sqrt(len(df_)))

mean_real=np.array(mean_real)
mean_reco=np.array(mean_reco)
mean_pnet=np.array(mean_pnet)
mean_part=np.array(mean_part)

std_real=np.array(std_real)
std_reco=np.array(std_reco)
std_pnet=np.array(std_pnet)
std_part=np.array(std_part)

#ax.errorbar(cuts, mean_pnet-mean_real,  yerr=std_pnet, linestyle='none', marker='o', label='pnet')
#ax.errorbar(cuts, mean_part-mean_real,  yerr=std_part, linestyle='none', marker='o', label='part')
#ax.errorbar(cuts, mean_reco-mean_real,  yerr=std_reco, linestyle='none', marker='o', label='reco')


x = cuts
ax.plot(x, mean_reco - mean_real,  label='reco', color='blue')
ax.plot(x, mean_pnet - mean_real, label='pnet', color='orange')
ax.plot(x, mean_part - mean_real, label='part', color='green')
#ax.hlines(y=125.09, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dotted', color='black')

# Add the shaded error regions

ax.fill_between(x, mean_reco - mean_real - std_reco, mean_reco - mean_real + std_reco, color='blue', alpha=0.2)
ax.fill_between(x, mean_pnet - mean_real - std_pnet, mean_pnet - mean_real + std_pnet, color='orange', alpha=0.2)
ax.fill_between(x, mean_part - mean_real - std_part, mean_part - mean_real + std_part, color='green', alpha=0.2)

ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dotted', color='black')
#ax.set_xlim(cuts[0], cuts[-1])
ax.set_xlabel("Jet1 & 2 pT [GeV]")
ax.set_ylabel("Dijet Mass Reco - GenDijet Mass [GeV]")

ax.legend()

# %%
# Check if the mean represents the position of the peak
c=cuts[19]
print(c)
mask = (df.jet1_pt>c) & (df.jet2_pt > c) & (abs(df.jet2_eta)<2.5) & (abs(df.jet1_eta)<2.5)
df_ = df[mask]

mean_real =  (np.median(df_.genDijet_mass))

mean_reco =  ( np.median(df_.dijet_mass_reco))
mean_pnet =  ( np.median(df_.dijet_mass_pnet))
mean_part =  ( np.median(df_.dijet_mass_parT))

std_reco = (np.std(df_.dijet_mass_reco)/np.sqrt(len(df_)))
std_pnet = (np.std(df_.dijet_mass_pnet)/np.sqrt(len(df_)))
std_part = (np.std(df_.dijet_mass_parT)/np.sqrt(len(df_)))

fig, ax = plt.subplots(1, 1)
bins=np.linspace(60, 180, 51)
c_reco  = ax.hist(df_.dijet_mass_reco, bins=bins, histtype='step', density=True)[0]
c_pnet = ax.hist(df_.dijet_mass_pnet, bins=bins, histtype='step', density=True)[0]
c_part = ax.hist(df_.dijet_mass_parT, bins=bins, histtype='step', density=True)[0]

ax.errorbar(x=mean_reco, y=np.max(c_reco), xerr=std_reco, marker='o', color='C0')
ax.errorbar(x=mean_pnet, y=np.max(c_pnet), xerr=std_pnet, marker='o', color='C1')
ax.errorbar(x=mean_part, y=np.max(c_part), xerr=std_part, marker='o', color='C2')








# %%






# Tagger


fig, ax = plt.subplots(1, 2, figsize=(20,10))
m = (abs(df.jet1_eta)<2.5) & (df.jet1_pt>0)
bins=np.linspace(0, 1, 11)
c_pnet = np.histogram(df[m].jet1_btagPNetB, bins=bins)[0]
c_parT = np.histogram(df[m].jet1_tagUParTAK4B, bins=bins)[0]
c_deep = np.histogram(df[m].jet1_btagDeepFlavB, bins=bins)[0]
c_pnet, c_parT, c_deep = c_pnet/np.sum(c_pnet), c_parT/np.sum(c_parT), c_deep/np.sum(c_deep)
ax[0].hist(bins[:-1], bins=bins, weights=c_deep, histtype='step', label='jet1_btagDeepFlavB')
ax[0].hist(bins[:-1], bins=bins, weights=c_pnet, histtype='step', label='jet1_btagPNetB')
ax[0].hist(bins[:-1], bins=bins, weights=c_parT, histtype='step', label='jet1_tagUParTAK4B')
ax[0].legend()

m = (abs(df.jet2_eta)<2.5) & (df.jet2_pt>0)
bins=np.linspace(0, 1, 11)
c_pnet = np.histogram(df[m].jet2_btagPNetB, bins=bins)[0]
c_parT = np.histogram(df[m].jet2_tagUParTAK4B, bins=bins)[0]
c_deep = np.histogram(df[m].jet2_btagDeepFlavB, bins=bins)[0]
c_pnet, c_parT, c_deep = c_pnet/np.sum(c_pnet), c_parT/np.sum(c_parT), c_deep/np.sum(c_deep)
ax[1].hist(bins[:-1], bins=bins, weights=c_deep, histtype='step', label='jet2_btagDeepFlavB')
ax[1].hist(bins[:-1], bins=bins, weights=c_pnet, histtype='step', label='jet2_btagPNetB')
ax[1].hist(bins[:-1], bins=bins, weights=c_parT, histtype='step', label='jet2_tagUParTAK4B')
ax[1].legend()

# %%

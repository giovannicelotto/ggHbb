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
df_jet1 = df[[col for col in df.columns if col.startswith('jet1') or col.startswith('genJetNu1')]]
df_jet2 = df[[col for col in df.columns if col.startswith('jet2') or col.startswith('genJetNu2')]]
df_jet1.columns = df_jet1.columns.str.replace('1', '', regex=False)
df_jet2.columns = df_jet2.columns.str.replace('2', '', regex=False)
#df = pd.concat([df_jet1, df_jet2])
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
binning = [20, 20, 30, 40, 50, 70]
for i in range(nrow):
    for j in range(ncol):
        bins = np.linspace(-binning[i*ncol+j], binning[i*ncol + j], 41)
        mask = (df.genJetNu1_pt > cuts[i*ncol + j]) & (df.genJetNu1_pt < cuts[i*ncol + j+1]) & (abs(df.genJetNu1_eta)<0.087)
        df_ = df[mask]
        print(len(df_))
        creg = ax[i,j].hist(np.clip(df_.jet1_pt*df_.jet1_bReg2018 - df_.genJetNu1_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='Regression')[0]
        cpnet = ax[i,j].hist(np.clip(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_PNetRegPtRawCorr * df_.jet1_PNetRegPtRawCorrNeutrino - df_.genJetNu1_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='PNet Neutrinos')[0]
        cpart = ax[i,j].hist(np.clip(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_ParTAK4RegPtRawCorr * df_.jet1_UParTAK4RegPtRawCorrNeutrino - df_.genJetNu1_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='ParT Neutrinos')[0]
        ax[i, j].set_title(" %d<GenJetNu 1 pT <%d"%(cuts[i*ncol + j], cuts[i*ncol + j+1]))

        ax[i, j].errorbar(x=np.median(df_.jet1_pt*df_.jet1_bReg2018 - df_.genJetNu1_pt), y=np.max(creg), marker='o', color='C0')
        ax[i, j].errorbar(x=np.median(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_PNetRegPtRawCorr * df_.jet1_PNetRegPtRawCorrNeutrino - df_.genJetNu1_pt), y=np.max(cpnet), marker='o', color='C1')
        ax[i, j].errorbar(x=np.median(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_ParTAK4RegPtRawCorr * df_.jet1_UParTAK4RegPtRawCorrNeutrino - df_.genJetNu1_pt), y=np.max(cpart), marker='o', color='C2')

ax[0, 0].legend()



# %%



cuts = np.logspace(1, np.log10(1000), 16)
bins = np.linspace(-70, 70, 101)

fig, ax = plt.subplots(1, 1)
mean_real = []
mean_raw = []
mean_reco = []
mean_reg = []
mean_pnet = []
mean_part = []
std_real = []
std_raw = []
std_reco = []
std_reg = []
std_pnet = []
std_part = []


df['jet1_pt_pnet'] = df.jet1_pt* (1-df.jet1_rawFactor) * df.jet1_PNetRegPtRawCorr * df.jet1_PNetRegPtRawCorrNeutrino
df['jet2_pt_pnet'] = df.jet2_pt* (1-df.jet2_rawFactor) * df.jet2_PNetRegPtRawCorr * df.jet2_PNetRegPtRawCorrNeutrino

for i in range(len(cuts)-1):
        mask = (df.genJetNu1_pt > cuts[i]) & (df.genJetNu1_pt < cuts[i+1]) & (abs(df.genJetNu1_eta)<1.3)
        df_ = df[mask]

        mean_real.append ( np.median(df_.genJetNu1_pt))

        mean_raw.append ( np.median(df_.jet1_pt* (1-df_.jet1_rawFactor) / df_.genJetNu1_pt))
        mean_reco.append ( np.median(df_.jet1_pt / df_.genJetNu1_pt))
        mean_reg.append ( np.median(df_.jet1_pt*df_.jet1_bReg2018 / df_.genJetNu1_pt))
        mean_pnet.append ( np.median(df_.jet1_pt_pnet / df_.genJetNu1_pt))
        mean_part.append ( np.median(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_ParTAK4RegPtRawCorr * df_.jet1_UParTAK4RegPtRawCorrNeutrino / df_.genJetNu1_pt))

        #std_real.append(np.std(df_.genJetNu1_pt))
        std_raw.append(np.std(df_.jet1_pt* (1-df_.jet1_rawFactor) / df_.genJetNu1_pt)/np.sqrt(len(df_)))
        std_reco.append(np.std(df_.jet1_pt / df_.genJetNu1_pt)/np.sqrt(len(df_)))
        std_reg.append(np.std(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_PNetRegPtRawCorr * df_.jet1_PNetRegPtRawCorrNeutrino / df_.genJetNu1_pt)/np.sqrt(len(df_)))
        std_pnet.append(np.std(df_.jet1_pt_pnet / df_.genJetNu1_pt)/np.sqrt(len(df_)))
        std_part.append(np.std(df_.jet1_pt* (1-df_.jet1_rawFactor) * df_.jet1_ParTAK4RegPtRawCorr * df_.jet1_UParTAK4RegPtRawCorrNeutrino / df_.genJetNu1_pt)/np.sqrt(len(df_)))

mean_real=np.array(mean_real)
mean_raw=np.array(mean_raw)
mean_reco=np.array(mean_reco)
mean_reg=np.array(mean_reg)
mean_pnet=np.array(mean_pnet)
mean_part=np.array(mean_part)
std_real=np.array(std_real)
std_raw=np.array(std_raw)
std_reco=np.array(std_reco)
std_reg=np.array(std_reg)
std_pnet=np.array(std_pnet)
std_part=np.array(std_part)

#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_raw, xerr=(cuts[1:]-cuts[:-1])/2, yerr=std_raw, linestyle='none', label='Raw')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_reco, xerr=(cuts[1:]-cuts[:-1])/2, yerr=std_reco, linestyle='none', label='Reco')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_reg, xerr=(cuts[1:]-cuts[:-1])/2, yerr=std_reg, linestyle='none', label='Reg')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_pnet,xerr=(cuts[1:]-cuts[:-1])/2,  yerr=std_pnet, linestyle='none', label='PNet')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_part,xerr=(cuts[1:]-cuts[:-1])/2,  yerr=std_part, linestyle='none', label='Part')
x = (cuts[1:] + cuts[:-1])/2

ax.plot(x, mean_reg, label='Reg', color='C0')
ax.plot(x, mean_pnet, label='PNetRegNeutrino', color='C1')
ax.plot(x, mean_part, label='ParTNeutrino', color='C2')

ax.text(x=0.1, y=0.1, s="|$\eta$|<1.3", transform=ax.transAxes)
#ax.plot(x, mean_real, label='gen', color='gray')
#
## Add the shaded error regions
#ax.fill_between(x, mean_real - std_real, mean_real + std_real, color='blue', alpha=0.2)
ax.fill_between(x, mean_reg - std_reg, mean_reg + std_reg, color='C0', alpha=0.2)
ax.fill_between(x, mean_pnet - std_pnet, mean_pnet + std_pnet, color='C1', alpha=0.2)
ax.fill_between(x, mean_part - std_part, mean_part + std_part, color='C2', alpha=0.2)


ax.hlines(y=1, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dotted', color='black')

ax.set_xlabel("Gen Jet1 pT")
ax.set_ylabel("Median Response ($p_T/p_T^{gen}$)")
ax.set_xscale('log')
ax.legend()
ax.set_xlim(x[0],x[-1])
ax.set_ylim(0.8,1.3)

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
        mask = (df.genJetNu2_pt > cuts[i*ncol + j]) & (df.genJetNu2_pt < cuts[i*ncol + j+1]) & (abs(df.jet2_eta)<2.5)
        df_ = df[mask]

        creg = ax[i,j].hist(np.clip(df_.jet2_pt*df_.jet2_bReg2018 - df_.genJetNu2_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='regression')[0]
        cpnet = ax[i,j].hist(np.clip(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_PNetRegPtRawCorr * df_.jet2_PNetRegPtRawCorrNeutrino - df_.genJetNu2_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='PNet Neutrinos')[0]
        cpart = ax[i,j].hist(np.clip(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr * df_.jet2_UParTAK4RegPtRawCorrNeutrino - df_.genJetNu2_pt, bins[0], bins[-1]), bins=bins, histtype='step', label='ParT Neutrinos')[0]
        
        ax[i, j].errorbar(x=np.median(df_.jet2_pt*df_.jet2_bReg2018 - df_.genJetNu2_pt), y=np.max(creg), marker='o', color='C0')
        ax[i, j].errorbar(x=np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_PNetRegPtRawCorr * df_.jet2_PNetRegPtRawCorrNeutrino - df_.genJetNu2_pt), y=np.max(cpnet), marker='o', color='C1')
        ax[i, j].errorbar(x=np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr * df_.jet2_UParTAK4RegPtRawCorrNeutrino - df_.genJetNu2_pt), y=np.max(cpart), marker='o', color='C2')


        print(np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr * df_.jet2_UParTAK4RegPtRawCorrNeutrino - df_.genJetNu2_pt))

        ax[i, j].set_title(" %d<GenJetNu 2 pT <%d"%(cuts[i*ncol + j], cuts[i*ncol + j+1]))

ax[0, 0].legend()


# %%



cuts = np.logspace(1, np.log10(300), 16)
bins = np.linspace(-70, 70, 101)

fig, ax = plt.subplots(1, 1)
mean_real = []
mean_raw = []
mean_reco = []
mean_reg = []
mean_pnet = []
mean_part = []
std_real = []
std_raw = []
std_reco = []
std_reg = []
std_pnet = []
std_part = []


for i in range(len(cuts)-1):
        mask = (df.genJetNu2_pt > cuts[i]) & (df.genJetNu2_pt < cuts[i+1]) & (abs(df.jet2_eta)<1.3)
        df_ = df[mask]

        mean_real.append ( np.median(df_.genJetNu2_pt))

        mean_raw.append ( np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) / df_.genJetNu2_pt))
        mean_reco.append ( np.median(df_.jet2_pt / df_.genJetNu2_pt))
        mean_reg.append ( np.median(df_.jet2_pt*df_.jet2_bReg2018 / df_.genJetNu2_pt))
        mean_pnet.append ( np.median(df_.jet2_pt_pnet / df_.genJetNu2_pt))
        mean_part.append ( np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr * df_.jet2_UParTAK4RegPtRawCorrNeutrino / df_.genJetNu2_pt))

        #std_real.append(np.std(df_.genJetNu2_pt))
        std_raw.append(np.std(df_.jet2_pt* (1-df_.jet2_rawFactor) / df_.genJetNu2_pt)/np.sqrt(len(df_)))
        std_reco.append(np.std(df_.jet2_pt / df_.genJetNu2_pt)/np.sqrt(len(df_)))
        std_reg.append(np.std(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_PNetRegPtRawCorr * df_.jet2_PNetRegPtRawCorrNeutrino / df_.genJetNu2_pt)/np.sqrt(len(df_)))
        std_pnet.append(np.std(df_.jet2_pt_pnet / df_.genJetNu2_pt)/np.sqrt(len(df_)))
        std_part.append(np.std(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr * df_.jet2_UParTAK4RegPtRawCorrNeutrino / df_.genJetNu2_pt)/np.sqrt(len(df_)))

mean_real=np.array(mean_real)
mean_raw=np.array(mean_raw)
mean_reco=np.array(mean_reco)
mean_reg=np.array(mean_reg)
mean_pnet=np.array(mean_pnet)
mean_part=np.array(mean_part)
std_real=np.array(std_real)
std_raw=np.array(std_raw)
std_reco=np.array(std_reco)
std_reg=np.array(std_reg)
std_pnet=np.array(std_pnet)
std_part=np.array(std_part)

#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_raw, xerr=(cuts[1:]-cuts[:-1])/2, yerr=std_raw, linestyle='none', label='Raw')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_reco, xerr=(cuts[1:]-cuts[:-1])/2, yerr=std_reco, linestyle='none', label='Reco')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_reg, xerr=(cuts[1:]-cuts[:-1])/2, yerr=std_reg, linestyle='none', label='Reg')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_pnet,xerr=(cuts[1:]-cuts[:-1])/2,  yerr=std_pnet, linestyle='none', label='PNet')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_part,xerr=(cuts[1:]-cuts[:-1])/2,  yerr=std_part, linestyle='none', label='Part')
x = (cuts[1:] + cuts[:-1])/2

ax.plot(x, mean_reg, label='Reg', color='C0')
ax.plot(x, mean_pnet, label='PNetRegNeutrino', color='C1')
ax.plot(x, mean_part, label='ParTNeutrino', color='C2')
ax.text(x=0.1, y=0.1, s="|$\eta$|<1.3", transform=ax.transAxes)
#ax.plot(x, mean_real, label='gen', color='gray')
#
## Add the shaded error regions
#ax.fill_between(x, mean_real - std_real, mean_real + std_real, color='blue', alpha=0.2)
ax.fill_between(x, mean_reg - std_reg, mean_reg + std_reg, color='C0', alpha=0.2)
ax.fill_between(x, mean_pnet - std_pnet, mean_pnet + std_pnet, color='C1', alpha=0.2)
ax.fill_between(x, mean_part - std_part, mean_part + std_part, color='C2', alpha=0.2)


ax.hlines(y=1, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dotted', color='black')
ax.set_ylabel("Median Response ($p_T/p_T^{gen}$)")
ax.set_xlabel("Gen Jet2 pT")

ax.set_xscale('log')
ax.legend()
ax.set_xlim(x[0],x[-1])
ax.set_ylim(0.8,1.3)


# %%

mask = (abs(df.jet1_eta)<2.5) & (abs(df.jet2_eta)<2.5) & (df.genJetNu1_pt>0) & (df.genJetNu2_pt>0) 


fig, ax = plt.subplots(1, 1)
bins = np.linspace(60, 180, 41)
ax.set_title("|eta|<2.5 & pT>?")
ax.hist(np.clip(df.dijet_mass_2018[mask], bins[0], bins[-1]), bins=bins, histtype='step', label='reg')
ax.hist(np.clip(df.dijet_mass_pnetNu[mask], bins[0], bins[-1]), bins=bins, histtype='step', label='pnetNu')
ax.hist(np.clip(df.dijet_mass_partTNu[mask], bins[0], bins[-1]), bins=bins, histtype='step', label='parTNu')
ax.hist(np.clip(df.genDijetNu_mass[mask], bins[0], bins[-1]), bins=bins, histtype='step', label='GenDijet')
ax.legend()
# %%
# dijet mass in bin of pt

bins = np.linspace(50, 180, 41)
#cuts = np.linspace(0, 200, 20)
cuts = np.array([15, 25, 40, 60, 80, 100, 200])
fig, ax = plt.subplots(1, 1)
mean_real = []
mean_raw  = []
mean_reco = []
mean_reg  = []
mean_pnet = []
mean_part = []
std_real  = []
std_raw   = []
std_reco  = []
std_reg   = []
std_pnet  = []
std_part  = []
for c in cuts:
        mask = (df.jet1_pt > c) & (df.jet2_pt > c) & (abs(df.jet2_eta)<2.5) & (abs(df.jet1_eta)<2.5)
        df_ = df[mask]

        mean_real.append ( np.median(df_.genDijetNu_mass))

        mean_reco.append ( np.median(df_.dijet_mass_reco))
        mean_reg.append ( np.median(df_.dijet_mass_2018))
        mean_pnet.append ( np.median(df_.dijet_mass_pnetNu))
        mean_part.append ( np.median(df_.dijet_mass_partTNu))

        std_real.append(np.std(df_.genDijetNu_mass)/np.sqrt(len(df_)))
        std_reco.append(np.std(df_.dijet_mass_reco)/np.sqrt(len(df_)))
        std_reg.append(np.std(df_.dijet_mass_2018)/np.sqrt(len(df_)))
        std_pnet.append(np.std(df_.dijet_mass_pnetNu)/np.sqrt(len(df_)))
        std_part.append(np.std(df_.dijet_mass_partTNu)/np.sqrt(len(df_)))



mean_real=np.array(mean_real)
mean_reco=np.array(mean_reco)
mean_reg=np.array(mean_reg)
mean_pnet=np.array(mean_pnet)
mean_part=np.array(mean_part)
std_real=np.array(std_real)
std_reco=np.array(std_reco)
std_reg=np.array(std_reg)
std_pnet=np.array(std_pnet)
std_part=np.array(std_part)

#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_reg, xerr=(cuts[1:]-cuts[:-1])/2, yerr=std_reg, linestyle='none', label='reg')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_pnet,xerr=(cuts[1:]-cuts[:-1])/2,  yerr=std_pnet, linestyle='none', label='pnet')
#ax.errorbar((cuts[1:] + cuts[:-1])/2, mean_part,xerr=(cuts[1:]-cuts[:-1])/2,  yerr=std_part, linestyle='none', label='part')



x = cuts

ax.plot(x, mean_reg, label='reg', color='blue')
ax.plot(x, mean_pnet, label='pnet', color='orange')
ax.plot(x, mean_part, label='part', color='green')
ax.plot(x, mean_real, label='gen', color='gray')
#
## Add the shaded error regions
ax.fill_between(x, mean_real - std_real, mean_real + std_real, color='blue', alpha=0.2)
ax.fill_between(x, mean_reg - std_reg, mean_reg + std_reg, color='blue', alpha=0.2)
ax.fill_between(x, mean_pnet - std_pnet, mean_pnet + std_pnet, color='orange', alpha=0.2)
ax.fill_between(x, mean_part - std_part, mean_part + std_part, color='green', alpha=0.2)
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
mean_reg  = []
mean_pnet = []
mean_part = []
std_real  = []
std_raw   = []
std_reco  = []
std_reg   = []
std_pnet  = []
std_part  = []
cuts = np.linspace(0, 200, 20)
for c in cuts:
        #mask = (df.genJetNu1_pt > c) & (df.genJetNu2_pt > c) & (abs(df.jet2_eta)<2.5) & (abs(df.jet1_eta)<2.5)
        print(c)
        mask = (df.jet1_pt>c) & (df.jet2_pt > c) & (abs(df.jet2_eta)<2.5) & (abs(df.jet1_eta)<2.5)
        df_ = df[mask]

        mean_real.append (np.median(df_.genDijetNu_mass))

        mean_reco.append ( np.median(df_.dijet_mass_reco))
        mean_reg.append ( np.median(df_.dijet_mass_2018))
        mean_pnet.append ( np.median(df_.dijet_mass_pnetNu))
        mean_part.append ( np.median(df_.dijet_mass_partTNu))

        std_reco.append(np.std(df_.dijet_mass_reco)/np.sqrt(len(df_)))
        std_reg.append(np.std(df_.dijet_mass_2018)/np.sqrt(len(df_)))
        std_pnet.append(np.std(df_.dijet_mass_pnetNu)/np.sqrt(len(df_)))
        std_part.append(np.std(df_.dijet_mass_partTNu)/np.sqrt(len(df_)))

mean_real=np.array(mean_real)
mean_reco=np.array(mean_reco)
mean_reg=np.array(mean_reg)
mean_pnet=np.array(mean_pnet)
mean_part=np.array(mean_part)
std_real=np.array(std_real)
std_reco=np.array(std_reco)
std_reg=np.array(std_reg)
std_pnet=np.array(std_pnet)
std_part=np.array(std_part)

#ax.errorbar(cuts, mean_reg-mean_real,  yerr=std_reg, linestyle='none', marker='o', label='reg')
#ax.errorbar(cuts, mean_pnet-mean_real,  yerr=std_pnet, linestyle='none', marker='o', label='pnet')
#ax.errorbar(cuts, mean_part-mean_real,  yerr=std_part, linestyle='none', marker='o', label='part')
#ax.errorbar(cuts, mean_reco-mean_real,  yerr=std_reco, linestyle='none', marker='o', label='reco')


x = cuts
ax.plot(x, mean_reg - mean_real, label='reg', color='blue')
ax.plot(x, mean_pnet - mean_real, label='pnet', color='orange')
ax.plot(x, mean_part - mean_real, label='part', color='green')
#ax.hlines(y=125.09, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dotted', color='black')

# Add the shaded error regions

ax.fill_between(x, mean_reg - mean_real - std_reg, mean_reg - mean_real + std_reg, color='blue', alpha=0.2)
ax.fill_between(x, mean_pnet - mean_real - std_pnet, mean_pnet - mean_real + std_pnet, color='orange', alpha=0.2)
ax.fill_between(x, mean_part - mean_real - std_part, mean_part - mean_real + std_part, color='green', alpha=0.2)

ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dotted', color='black')
#ax.set_xlim(cuts[0], cuts[-1])
ax.set_xlabel("Jet1 & 2 pT [GeV]")
ax.set_ylabel("Dijet Mass Reco - GenDijet Mass [GeV]")

ax.legend()

# %%
# Check if the mean represents the position of the peak



# check tra 60 e 80

cuts = np.array([15, 25, 40,
        60, 80, 100, 200])
clow = 60
chigh = 60
print(c)
mask = (df.jet2_pt > clow) & (abs(df.jet2_eta)<2.5)
df_ = df[mask]

mean_reg=[]
mean_pnet=[]
mean_part=[]

#mean_reg.append ( np.median(df_.jet2_pt*df_.jet2_bReg2018 - df_.genJetNu2_pt))
#mean_pnet.append ( np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_PNetRegPtRawCorr * df_.jet2_PNetRegPtRawCorrNeutrino - df_.genJetNu2_pt))
#mean_part.append ( np.median(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr * df_.jet2_UParTAK4RegPtRawCorrNeutrino - df_.genJetNu2_pt))
mean_reg.append(np.median(df_.dijet_mass_2018 - df_.genDijetNu_mass))
mean_pnet.append(np.median(df_.dijet_mass_pnetNu - df_.genDijetNu_mass))
mean_part.append(np.median(df_.dijet_mass_partTNu - df_.genDijetNu_mass))


fig, ax = plt.subplots(1, 1)
bins=np.linspace(-50, 50, 51)
#c_reg  = ax.hist(df_.jet2_pt*df_.jet2_bReg2018 - df_.genJetNu2_pt, bins=bins, histtype='step', density=True)[0]
#c_pnet = ax.hist(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_PNetRegPtRawCorr * df_.jet2_PNetRegPtRawCorrNeutrino - df_.genJetNu2_pt, bins=bins, histtype='step', density=True)[0]
#c_part = ax.hist(df_.jet2_pt* (1-df_.jet2_rawFactor) * df_.jet2_ParTAK4RegPtRawCorr * df_.jet2_UParTAK4RegPtRawCorrNeutrino - df_.genJetNu2_pt, bins=bins, histtype='step', density=True)[0]
c_reg  = ax.hist(df_.dijet_mass_2018 - df_.genDijetNu_mass, bins=bins, histtype='step', density=True)[0]
c_pnet = ax.hist(df_.dijet_mass_pnetNu - df_.genDijetNu_mass, bins=bins, histtype='step', density=True)[0]
c_part = ax.hist(df_.dijet_mass_partTNu - df_.genDijetNu_mass, bins=bins, histtype='step', density=True)[0]

ax.errorbar(x=mean_reg, y=np.max(c_reg),  marker='o', color='C0')
ax.errorbar(x=mean_pnet, y=np.max(c_pnet),  marker='o', color='C1')
ax.errorbar(x=mean_part, y=np.max(c_part),  marker='o', color='C2')






# %%
fig, ax = plt.subplots(1, 3, figsize=(20, 8))
bins=np.linspace(0, 150, 51)
ax[0].hist(df.genJetNu1_pt, bins=bins, histtype='step', density=True)
ax[0].hist(df.genJetNu2_pt, bins=bins, histtype='step', density=True)
ax[0].set_xlabel("GenJet pT")
bins=np.linspace(-5, 5, 51)
ax[1].hist(df.jet1_eta, bins=bins, histtype='step', density=True)
ax[1].hist(df.jet2_eta[df.jet2_pt>0], bins=bins, histtype='step', density=True)
ax[1].set_xlabel("Jet eta")
bins=np.linspace(0, .1, 51)
ax[2].hist(df.jet1_genJetNu_dR, bins=bins, histtype='step', density=True)
ax[2].hist(df.jet2_genJetNu_dR, bins=bins, histtype='step', density=True)
ax[2].set_xlabel("GenJet dR GenJetNu")

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

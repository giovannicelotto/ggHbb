# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from functions import getCommonFilters
# %%
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Mar22_11_50p0/dataframes_Data1D_Mar22_11_50p0.parquet"
path2 = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Mar22_11_50p0/dataframes_Data2D_Mar22_11_50p0.parquet"
path3 = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Mar22_11_50p0/dataframes_Data3D_Mar22_11_50p0.parquet"
df = pd.read_parquet([path, path2, path3])
# %%
import glob
df = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1D/*.parquet")[:100], filters=getCommonFilters(btagWP="M", cutDijet=False))
df = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLO*/others/*.parquet")[:100], filters=getCommonFilters(btagWP="M", cutDijet=False))
nn_cut = 0.9
mass_cut =50
dr_cut = 1.
bins = np.linspace(0, 300, 101)
#df = df[abs(df.dijet_pT_asymmetry) < 0.45]

# %%
fig, ax = plt.subplots(1, 1)

ax.hist(df.dijet_mass[(df.dijet_pt>60) & (df.dijet_pt<120) ], bins=bins, histtype='step', density=False,label='60<pt(bb)<120 GeV')
ax.hist(df.dijet_mass[(df.dijet_pt>60) & (df.dijet_pt<120) & (abs(df.dijet_pT_asymmetry)<0.45) ], bins=bins, histtype='step', density=False, label='+ |pT asym| < 0.45')

ax.text(x=0.6, y=0.9, s="Efficiency: %.1f%%"%(100*len(df[(df.dijet_pt>60) & (df.dijet_pt<120) & (abs(df.dijet_pT_asymmetry)<0.45)])/len(df[(df.dijet_pt>60) & (df.dijet_pt<120)])), transform=ax.transAxes)
#ax.hist(df.dijet_mass[(df.dijet_pt>60) & (abs(df.dijet_pT_asymmetry)<0.45)], bins=bins, histtype='step', density=False, color='black', label='pt>60 e asym')
#ax.hist(df.dijet_mass[(df.dijet_pt>60) & (df.dijet_pt<120) & (abs(df.dijet_pT_asymmetry)<0.45)], bins=bins, histtype='step', density=False, color='red', label='120>pt>60 e asym')
#ax.hist(df.dijet_mass[((df.dijet_pt/df.dijet_mass)>0.48) & ((df.dijet_pt/df.dijet_mass)<1.5) & (abs(df.dijet_pT_asymmetry)<0.45)], bins=bins, histtype='step', density=False, color='red', label='pt_prime>0.48 & pt_prime<1.5')
#ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_pt/df.dijet_mass<1)], bins=bins, label=f'nn> {nn_cut} pt_prime<1', histtype='step', density=True)
#ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_pt/df.dijet_mass<0.5)], bins=bins, label=f'nn> {nn_cut} pt_prime<0.5', histtype='step', density=True)
#c=ax.hist(df.dijet_mass[(df.PNN>nn_cut) & ((df.dijet_pt/df.dijet_mass)<2)], bins=bins, label=f'nn> {nn_cut} pt_prime<2', alpha=0.3, density=False)[0]
#ax.hist(df.dijet_mass[(df.PNN>nn_cut) & ((df.dijet_pt/df.dijet_mass)>2)], bins=bins, label=f'nn> {nn_cut} pt_prime>2', alpha=0.3, density=False, bottom=c)
ax.legend()
ax.set_ylabel("Events")
ax.set_xlabel("dijet mass")

# %%
fig, ax = plt.subplots(1, 1)
#ax.hist(df.dijet_mass, bins=bins, label='asym cut', histtype='step', density=True)
ax.hist(df.dijet_mass[df.PNN>nn_cut], bins=bins, label=f'nn> {nn_cut}', histtype='step', density=False, color='black')
#ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_pt/df.dijet_mass<1)], bins=bins, label=f'nn> {nn_cut} pt_prime<1', histtype='step', density=True)
#ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_pt/df.dijet_mass<0.5)], bins=bins, label=f'nn> {nn_cut} pt_prime<0.5', histtype='step', density=True)
c=ax.hist(df.dijet_mass[(df.PNN>nn_cut) & ((df.dijet_pt/df.dijet_mass)<2)], bins=bins, label=f'nn> {nn_cut} pt_prime<2', alpha=0.3, density=False)[0]
ax.hist(df.dijet_mass[(df.PNN>nn_cut) & ((df.dijet_pt/df.dijet_mass)>2)], bins=bins, label=f'nn> {nn_cut} pt_prime>2', alpha=0.3, density=False, bottom=c)
ax.legend()
ax.set_xlabel("dijet mass")

fig, ax = plt.subplots(1, 1)
#ax.hist(df.dijet_mass, bins=bins, label='asym cut', histtype='step', density=True)
ax.hist(df.dijet_mass[df.PNN>nn_cut], bins=bins, label=f'nn> {nn_cut}', histtype='step', density=False, color='black')
#ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_pt/df.dijet_mass<1)], bins=bins, label=f'nn> {nn_cut} pt_prime<1', histtype='step', density=True)
#ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_pt/df.dijet_mass<0.5)], bins=bins, label=f'nn> {nn_cut} pt_prime<0.5', histtype='step', density=True)
c=ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_dR>dr_cut)], bins=bins, label=f'nn> {nn_cut} dR>{dr_cut}', alpha=0.3, density=False)[0]
ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_dR<dr_cut)], bins=bins, label=f'nn> {nn_cut} dR<{dr_cut}', alpha=0.3, density=False, bottom=c)
ax.legend()
ax.set_xlabel("dijet mass")


# %%
fig, ax = plt.subplots(1, 1)


# --- parameters ---
k = 2.4           # tune this

# --- compute dynamic threshold ---
dr_dynamic = k * df.dijet_mass / df.dijet_pt

# --- masks ---
mask_nn = df.PNN > nn_cut
mask_param = (mask_nn) & (df.dijet_dR > dr_dynamic)

# --- plots ---
ax.hist(df.dijet_mass, bins=bins, density=True,
        histtype='step', label="no mask")

ax.hist(df.dijet_mass[mask_nn], bins=bins, density=True,
        histtype='step', label=f"NN > {nn_cut}")

ax.hist(df.dijet_mass[mask_param], bins=bins, density=True,
        histtype='step', label=f"NN > {nn_cut}, dR > k*mH/pT")

ax.legend()
ax.set_xlabel('dijet_mass')
# %%

fig, ax = plt.subplots(1,1 )
bins = np.linspace(0, 300, 101)
ax.hist(df.dijet_mass, bins=bins, density=True, histtype='step', label="no mask")[0]
ax.hist(df.dijet_mass[df.PNN>nn_cut], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut}")[0]
#ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_dR>dr_cut)], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut} dR>{dr_cut}")[0]
ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_dR>0.8)], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut} dR>0.8")[0]
ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_dR>1)], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut} dR>1")[0]
#ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_dR>1.05)], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut} dR>1.05")[0]
ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_dR>1.1)], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut} dR>1.1")[0]
ax.legend()
ax.set_xlabel('dijet_mass')
# %%
fig, ax = plt.subplots(1,1 )
bins = np.linspace(0, 300, 101)
#ax.hist(df.dijet_mass, bins=bins, density=True, histtype='step', label="no mask")[0]
ax.hist(df.dijet_mass[df.PNN>nn_cut], bins=bins, density=False, histtype='step', label=f"NN>{nn_cut}")[0]
ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_dR>dr_cut)], bins=bins, density=False, histtype='step', label=f"NN>{nn_cut} dR>{dr_cut}")[0]
ax.legend()
ax.set_xlabel('dijet_mass')

# %%
fig, ax = plt.subplots(1, 1)

bins = np.linspace(0, 300, 100)

mask = df.PNN > 0.85

ax.hist(df.dijet_mass[mask & (abs(df.dijet_pT_asymmetry) < 0.05)], bins=bins,histtype='step', density=True, label='dijet_pT_asymmetry < 0.2')
ax.hist(df.dijet_mass[mask & (abs(df.dijet_pT_asymmetry) > 0.4)], bins=bins,histtype='step', density=True, label='dijet_pT_asymmetry > 0.4')

ax.legend()
# %%

fig, ax = plt.subplots(1,1 )
bins = np.linspace(0, 300, 101)
ax.hist(df.dijet_pt, bins=bins, density=True, histtype='step', label="no mask")[0]
ax.hist(df.dijet_pt[df.PNN>nn_cut], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut}")[0]
ax.hist(df.dijet_pt[(df.PNN>nn_cut) & (df.dijet_dR>dr_cut)], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut} dR>0.8")[0]
ax.legend()
ax.set_xlabel('dijet_pt')
# %%

fig, ax = plt.subplots(1,1 )
bins = np.linspace(0, 3, 101)
ax.hist(df.dijet_dR, bins=bins, density=True, histtype='step', label="no mask")[0]
ax.hist(df.dijet_dR[df.PNN>nn_cut], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut}")[0]
ax.hist(df.dijet_dR[(df.PNN>nn_cut) & (df.dijet_dR>dr_cut)], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut} dR>0.8")[0]
ax.legend()
ax.set_xlabel('deltaR')
# %%
fig, ax = plt.subplots(1,1 )
bins = np.linspace(0, 300, 101)

ax.hist(df.dijet_mass[df.PNN>nn_cut], bins=bins, density=False, histtype='step', label=f"NN>{nn_cut}", linewidth=3)[0]
ax.hist(df.dijet_mass[(df.PNN>nn_cut) & (df.dijet_dR>dr_cut)], bins=bins, density=False, histtype='step', label=f"NN>{nn_cut} dR>0.8")[0]
ax.legend()
ax.set_xlabel('dijet_mass')

# %%
fig, ax = plt.subplots(1,1 )
bins = np.linspace(0, 300, 101)
ax.hist(df.dijet_pt, bins=bins, density=True, histtype='step', label="no mask")[0]
ax.hist(df.dijet_pt[df.PNN>nn_cut], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut}")[0]
ax.hist(df.dijet_pt[(df.PNN>nn_cut) & (df.dijet_dR>0.8)], bins=bins, density=True, histtype='step', label=f"NN>{nn_cut} dR>0.8")[0]
ax.legend()
ax.set_xlabel('dijet_pt')
# %%
mask = (df.PNN>nn_cut)  & (df.dijet_mass<mass_cut)
mask2 = (df.PNN>nn_cut)  & (df.dijet_mass>mass_cut)
# plot all features with this mask in a grid and in same binning the features with mask2
features = [col for col in df.columns if col not in ['dijet_mass', 'PNN']]
fig, axs = plt.subplots(10, 4, figsize=(20, 40))
for i, feature in enumerate(features):
    ax = axs[i//4, i%4]
    bins = np.linspace(df[feature].min(), df[feature].max(), 50)
    ax.hist(df[feature][mask], bins=bins, density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass<{mass_cut}')
    ax.hist(df[feature][mask2], bins=bins, density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass>{mass_cut}')
    ax.set_title(feature)
    ax.legend()
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(df.jet1_pt[mask], bins=np.linspace(0, 300, 100), density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass<{mass_cut}')
ax.hist(df.jet1_pt[mask2], bins=np.linspace(0, 300, 100), density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass>{mass_cut}')
ax.set_xlabel('jet1_pt')
ax.legend()

# %%
fig, ax = plt.subplots(1, 1)
ax.hist(df.dR_jet2_dijet[mask], bins=np.linspace(0, 4, 50), density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass<{mass_cut}')
ax.hist(df.dR_jet2_dijet[mask2], bins=np.linspace(0, 4, 50), density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass>{mass_cut}')
ax.set_xlabel('dR_jet2_dijet')
ax.legend()

# compute df.dijet_dR
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(df.dijet_dR[mask], bins=np.linspace(0, 3, 50), density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass<{mass_cut}')
ax.hist(df.dijet_dR[mask2], bins=np.linspace(0, 3, 50), density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass>{mass_cut}')
ax.set_xlabel('dijet_dr')
ax.legend()
# %%

fig, ax = plt.subplots(1, 1)
ax.hist(df.dijet_twist[mask], bins=np.linspace(0, 2, 50), density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass<{mass_cut}')
ax.hist(df.dijet_twist[mask2], bins=np.linspace(0, 2, 50), density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass>{mass_cut}')
ax.set_xlabel('dijet_twist')
ax.legend()

# %%
fig, ax = plt.subplots(1,1 )
bins = np.linspace(0, 300, 101)
ax.hist(df.dijet_mass, bins=bins, density=True, histtype='step')[0]
ax.hist(df.dijet_mass[(df.PNN>nn_cut) & ~((df.dR_jet2_dijet < 0.4) & (df.dR_jet1_dijet < 0.4))], bins=bins, density=True, histtype='step')[0]
# %%



mask = (df.PNN>nn_cut)  & (df.dijet_mass<60) & (df.dijet_dR>0.8)
mask2 = (df.PNN>nn_cut)  & (df.dijet_mass>60) & (df.dijet_dR>0.8)
# plot all features with this mask in a grid and in same binning the features with mask2
features = [col for col in df.columns if col not in ['dijet_mass', 'PNN']]
fig, axs = plt.subplots(10, 4, figsize=(20, 40))
for i, feature in enumerate(features):
    ax = axs[i//4, i%4]
    bins = np.linspace(df[feature].min(), df[feature].max(), 50)
    ax.hist(df[feature][mask], bins=bins, density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass<{mass_cut}')
    ax.hist(df[feature][mask2], bins=bins, density=True, histtype='step', label=f'PNN>{nn_cut} & dijet_mass>{mass_cut}')
    ax.set_title(feature)
    ax.legend()
# %%

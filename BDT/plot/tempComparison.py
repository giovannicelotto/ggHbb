# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import glob


# 1) GluGluHToBB New vs Old Jet Selection. No pT cut
df_old = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/flat_old/GluGluHToBB_flat_old/training/*.parquet"))
df_new = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training/*.parquet"))

fig, ax =plt.subplots(1, 1)
bins=np.linspace(0, 200, 101)
# plotting normalized 
c_old = np.histogram(np.clip(df_old.dijet_mass, bins[0], bins[-1]), bins=bins)[0]
c_new = np.histogram(np.clip(df_new.dijet_mass, bins[0], bins[-1]), bins=bins)[0]
c_old = c_old/np.sum(c_old)
c_new = c_new/np.sum(c_new)
ax.hist(bins[:-1], bins=bins, weights=c_old, histtype=u'step', density=True, color='C0', label='Old btag-based')
ax.hist(bins[:-1], bins=bins, weights=c_new, histtype=u'step', density=True, color='C1', label='New bdt-based')

# Info on the datasets
ax.text(x=0,y=0.015,  s="Std : %.1f"%(df_old.dijet_mass.std()), color='C0')
ax.text(x=0,y=0.014,  s="Events : %d"%(len(df_old)), color='C0')
ax.text(x=0,y=0.01,  s="Std : %.1f"%(df_new.dijet_mass.std()), color='C1')
ax.text(x=0,y=0.009,  s="Events : %d"%(len(df_new)), color='C1')

ax.text(x=0,y=0.005,  s="No Jet pT cut", color='black')

ax.set_ylim(ax.get_ylim())
ax.vlines(x=[90, 150], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='red')
old_sigEv=((df_old.dijet_mass>90) & (df_old.dijet_mass<150)).sum()
new_sigEv=((df_new.dijet_mass>90) & (df_new.dijet_mass<150)).sum()

print(new_sigEv/old_sigEv)
ax.legend(loc='best')

# %%

# 2) GluGluHToBB New vs Old Jet Selection.  pT > 20

fig, ax =plt.subplots(1, 1)
bins=np.linspace(0, 200, 101)
m_old = (df_old.jet1_pt>20) & (df_old.jet2_pt>20)
m_new = (df_new.jet1_pt>20) & (df_new.jet2_pt>20)
c_old = np.histogram(np.clip(df_old.dijet_mass[m_old], bins[0], bins[-1]), bins=bins)[0]
c_new = np.histogram(np.clip(df_new.dijet_mass[m_new], bins[0], bins[-1]), bins=bins)[0]
c_old = c_old/np.sum(c_old)
c_new = c_new/np.sum(c_new)
ax.hist(bins[:-1], bins=bins, weights=c_old, histtype=u'step', density=True, color='C0', label='Old btag-based')
ax.hist(bins[:-1], bins=bins, weights=c_new, histtype=u'step', density=True, color='C1', label='New bdt-based')

ax.text(x=0,y=0.015,  s="Std : %.1f"%(df_old.dijet_mass[m_old].std()), color='C0')
ax.text(x=0,y=0.014,  s="Events : %d"%(len(df_old[m_old])), color='C0')
ax.text(x=0,y=0.01,  s="Std : %.1f"%(df_new.dijet_mass[m_new].std()), color='C1')
ax.text(x=0,y=0.009,  s="Events : %d"%(len(df_new[m_new])), color='C1')

ax.text(x=0,y=0.005,  s="Jets pT > 20 GeV", color='black')

ax.set_ylim(ax.get_ylim())
ax.vlines(x=[90, 150], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='red')
old_sigEv=((df_old[m_old].dijet_mass>90) & (df_old[m_old].dijet_mass<150)).sum()
new_sigEv=((df_new[m_new].dijet_mass>90) & (df_new[m_new].dijet_mass<150)).sum()

print(new_sigEv/old_sigEv)
ax.legend(loc='best')

# %%




# Data










# %%
l_new = []
l_old = []
for i in [13, 14, 15, 16, 17]:
    l_new.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training/Data_%d.parquet"%i)
    l_old.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A_old/others/Data_%d.parquet"%i)
# %%
df_old = pd.read_parquet(l_old)
df_new = pd.read_parquet(l_new)


# %%
# 1) Data New vs Old Jet Selection. No pT cut

fig, ax =plt.subplots(1, 1)
bins=np.linspace(0, 250, 101)
# plotting normalized 
c_old = np.histogram(df_old.dijet_mass, bins=bins)[0]
c_new = np.histogram(df_new.dijet_mass, bins=bins)[0]
c_old = c_old/np.sum(c_old)
c_new = c_new/np.sum(c_new)
ax.hist(bins[:-1], bins=bins, weights=c_old, histtype=u'step', density=True, color='C0', label='Old btag-based')
ax.hist(bins[:-1], bins=bins, weights=c_new, histtype=u'step', density=True, color='C1', label='New bdt-based')

# Info on the datasets
ax.text(x=0.7,y=0.85,  s="Std : %.1f"%(df_old.dijet_mass.std()), color='C0', transform=ax.transAxes)
ax.text(x=0.7,y=0.8,  s="Events : %d"%(len(df_old)), color='C0', transform=ax.transAxes)
ax.text(x=0.7,y=0.5,  s="Std : %.1f"%(df_new.dijet_mass.std()), color='C1', transform=ax.transAxes)
ax.text(x=0.7,y=0.45,  s="Events : %d"%(len(df_new)), color='C1', transform=ax.transAxes)

ax.text(x=0.7,y=0.1,  s="No Jet pT cut", color='black', transform=ax.transAxes)

ax.set_ylim(ax.get_ylim())
#ax.vlines(x=[90, 150], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='red')
old_sigEv=((df_old.dijet_mass>90) & (df_old.dijet_mass<150)).sum()
new_sigEv=((df_new.dijet_mass>90) & (df_new.dijet_mass<150)).sum()

print(new_sigEv/old_sigEv)
ax.legend(loc='best')

# %%

# 2) Data New vs Old Jet Selection.  pT > 20


fig, ax =plt.subplots(1, 1)
bins=np.linspace(0, 250, 101)
m_old = (df_old.jet1_pt>20) & (df_old.jet2_pt>20)
m_new = (df_new.jet1_pt>20) & (df_new.jet2_pt>20)
c_old = np.histogram(df_old.dijet_mass[m_old], bins=bins)[0]
c_new = np.histogram(df_new.dijet_mass[m_new], bins=bins)[0]
c_old = c_old/np.sum(c_old)
c_new = c_new/np.sum(c_new)
ax.hist(bins[:-1], bins=bins, weights=c_old, histtype=u'step', density=True, color='C0', label='Old btag-based')
ax.hist(bins[:-1], bins=bins, weights=c_new, histtype=u'step', density=True, color='C1', label='New bdt-based')

# Info on the datasets

ax.text(x=0.7,y=0.85,   s="Std : %.1f"%(df_old.dijet_mass[m_old].std()), color='C0', transform=ax.transAxes)
ax.text(x=0.7,y=0.8,    s="Events : %d"%(len(df_old[m_old])), color='C0', transform=ax.transAxes)
ax.text(x=0.7,y=0.5,   s="Std : %.1f"%(df_new.dijet_mass[m_new].std()), color='C1', transform=ax.transAxes)
ax.text(x=0.7,y=0.45,   s="Events : %d"%(len(df_new[m_new])), color='C1', transform=ax.transAxes)

ax.text(x=0.7,y=0.3,   s="Jets pT > 20 GeV", color='black', transform=ax.transAxes)

ax.set_ylim(ax.get_ylim())
#ax.vlines(x=[90, 150], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='red')
old_sigEv=((df_old[m_old].dijet_mass>90) & (df_old[m_old].dijet_mass<150)).sum()
new_sigEv=((df_new[m_new].dijet_mass>90) & (df_new[m_new].dijet_mass<150)).sum()

print(new_sigEv/old_sigEv)
ax.legend(loc='best')
# %%

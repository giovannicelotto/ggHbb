# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import glob

df_old = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/*.parquet"))
df_new = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/GluGluHToBB_19*.parquet"))

# %%
fig, ax =plt.subplots(1, 1)
bins=np.linspace(0, 200, 101)
c_old = np.histogram(np.clip(df_old.dijet_mass, bins[0], bins[-1]), bins=bins)[0]
c_new = np.histogram(np.clip(df_new.dijet_mass, bins[0], bins[-1]), bins=bins)[0]
c_old = c_old/np.sum(c_old)
c_new = c_new/np.sum(c_new)
ax.hist(bins[:-1], bins=bins, weights=c_old, histtype=u'step', density=True, color='C0', label='Old btag-based')
ax.hist(bins[:-1], bins=bins, weights=c_new, histtype=u'step', density=True, color='C1', label='New bdt-based')
ax.text(x=0,y=0.015,  s="Std : %.1f"%(df_old.dijet_mass.std()), color='C0')
ax.text(x=0,y=0.014,  s="Events : %d"%(len(df_old)), color='C0')


ax.text(x=0,y=0.01,  s="Std : %.1f"%(df_new.dijet_mass.std()), color='C1')
ax.text(x=0,y=0.009,  s="Events : %d"%(len(df_new)), color='C1')
#ax.text(x=200,y=0.008, s="Std : %.1f"%(df_new.dijet_mass.std()), color='C1')
print("Length df old", len(df_old))

ax.legend(loc='best')
# %%


# %%


df_old = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/Data_44.parquet"))
df_new = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A_old/others/Data_44.parquet"))

# %%
fig, ax =plt.subplots(1, 1)
bins=np.linspace(0, 200, 101)
c_old = np.histogram(np.clip(df_old.dijet_mass, bins[0], bins[-1]), bins=bins)[0]
c_new = np.histogram(np.clip(df_new.dijet_mass, bins[0], bins[-1]), bins=bins)[0]
c_old = c_old/np.sum(c_old)
c_new = c_new/np.sum(c_new)
ax.hist(bins[:-1], bins=bins, weights=c_old, histtype=u'step', density=True, color='C0', label='Old btag-based')
ax.hist(bins[:-1], bins=bins, weights=c_new, histtype=u'step', density=True, color='C1', label='New bdt-based')
ax.text(x=0,y=0.015,  s="Std : %.1f"%(df_old.dijet_mass.std()), color='C0')
ax.text(x=0,y=0.014,  s="Events : %d"%(len(df_old)), color='C0')


ax.text(x=0,y=0.01,  s="Std : %.1f"%(df_new.dijet_mass.std()), color='C1')
ax.text(x=0,y=0.009,  s="Events : %d"%(len(df_new)), color='C1')
#ax.text(x=200,y=0.008, s="Std : %.1f"%(df_new.dijet_mass.std()), color='C1')
print("Length df old", len(df_old))

ax.legend(loc='best')
# %%


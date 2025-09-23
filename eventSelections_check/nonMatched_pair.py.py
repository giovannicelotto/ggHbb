# %%
import numpy as np
import pandas as pd
import glob
# %%
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/MC/MINLOGluGluHToBB/others/*.parquet")
df = pd.read_parquet(fileNames)
# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.hist(df.dijet_mass[abs(df.GenJet1_partonMotherPdgId)==5], bins=np.linspace(50, 200, 20))

# %%
print(np.sum(df.GenJet1_partonMotherPdgId==25)/len(df))
print(np.sum(df.GenJet2_partonMotherPdgId==25)/len(df))
print(np.sum((df.GenJet1_partonMotherPdgId==25) & (df.GenJet2_partonMotherPdgId==25))/len(df))
# %%
print(np.sum((df.GenJet1_partonMotherPdgId==25) | (df.GenJet1_partonMotherPdgId==5))/len(df))
print(np.sum((df.GenJet2_partonMotherPdgId==25) | (df.GenJet2_partonMotherPdgId==5))/len(df))
print(np.sum(((df.GenJet1_partonMotherPdgId==25) & (df.GenJet2_partonMotherPdgId==25)) | 
             ((df.GenJet2_partonMotherPdgId==5) & (df.GenJet2_partonMotherPdgId==5)))/len(df))
# %%

# %%
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
import glob
import numpy as np
# %%
fileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1D/training/*.parquet")
filters = [
            [   ('jet1_pt', '>',  20),
                ('jet2_pt', '>',  20),
                
                ('jet1_mass', '>', 0),
                ('jet2_mass', '>', 0),
                #('jet3_mass', '>',  0),
                
                ('jet1_eta', '>', -2.5),
                ('jet2_eta', '>', -2.5),
                ('jet1_eta', '<',  2.5),
                ('jet2_eta', '<',  2.5),
                #('jet1_btagDeepFlavB', '<',  0.71),
                ('jet2_btagDeepFlavB', '>',  0.71),
                ('muon_pt', '>=',  9.0),
                ('muon_eta', '>=',  -1.5),
                ('muon_eta', '<=',  1.5),
                ('muon_dxySig', '>=', 6.0)
                ],
                  
                  # OR Condition

            [   ('jet1_pt', '>',  20),
                ('jet2_pt', '>',  20),
                
                ('jet1_mass', '>', 0),
                ('jet2_mass', '>', 0),
                
                ('jet1_eta', '>', -2.5),
                ('jet2_eta', '>', -2.5),
                ('jet1_eta', '<',  2.5),
                ('jet2_eta', '<',  2.5),
                #('jet1_btagDeepFlavB', '<',  0.71),
                ('jet2_btagDeepFlavB', '>',  0.71),
                ('muon_pt', '>=',  9.0),
                ('muon_eta', '>=',  -1.5),
                ('muon_eta', '<=',  1.5),
                ('muon_dxySig', '<=', -6.0)
                ]

    ]
df =pd.read_parquet(fileNames, filters=filters)
# %%
# %%


# Dividi il DataFrame in due categorie
df_high = df[df['jet1_btagDeepFlavB'] >= 0.71]
df_low  = df[(df['jet1_btagDeepFlavB'] < 0.71) & (df['jet1_btagDeepFlavB'] >= 0.2783)]

# Escludi colonne non numeriche se necessario
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Setup subplot grid
n_cols = 4
n_rows = int(np.ceil(len(numeric_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
axes = axes.flatten()

# Loop su tutte le colonne numeriche
for i, col in enumerate(numeric_cols):
    ax = axes[i]
    #if np.min(df_high[col])==-999:
        
    ax.hist(df_high[col], bins=30, alpha=0.5, label='≥ 0.71', color='red', density=True)
    ax.hist(df_low[col], bins=30, alpha=0.5, label='< 0.71', color='blue', density=True)
    ax.set_title(col)
    ax.legend()

# Rimuove subplot inutilizzati
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
# %%

# %%
import numpy as np
import matplotlib.pyplot as plt
import json, sys, glob, re
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.preprocessMultiClass import preprocessMultiClass
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helpersABCD.loadDataFrames import loadDataFrames
import dcor
# %%

nReal = 10
nMC = 1
predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18"
columns = ['dijet_pt',           'dijet_mass', 
          'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
          'dijet_cs',   'leptonClass',          
          'PU_SF', 'sf', 
          ]
columns = None

dfs, isMCList, dfProcesses, nReal = loadDataFrames(nReal=nReal, nMC=nMC, predictionsPath=predictionsPath, columns=columns)
for idx, df in enumerate(dfs):
    dfs[idx]['dijet_cs_abs'] = 1-abs(dfs[idx].dijet_cs)

# %%

def distance_correlation_table(df, n_events=1000):
    """
    Computes a pairwise distance correlation table for the first n_events of the DataFrame using dcor.
    """
    df_subset = df.iloc[:n_events]
    columns = df_subset.columns
    n_cols = len(columns)
    
    # Initialize an empty correlation matrix
    dist_corr_matrix = np.zeros((n_cols, n_cols))
    
    # Compute distance correlation for each pair of columns
    for i in range(n_cols):
        print(i,  "/", n_cols)
        for j in range(i, n_cols):
            corr = dcor.distance_correlation(df_subset.iloc[:, i].values, df_subset.iloc[:, j].values)
            dist_corr_matrix[i, j] = corr
            dist_corr_matrix[j, i] = corr  # Symmetric matrix
    
    # Create a DataFrame for the correlation matrix
    return pd.DataFrame(dist_corr_matrix, index=columns, columns=columns)


# Compute the distance correlation table
dist_corr_table = distance_correlation_table(dfs[0], n_events=1000)
print(dist_corr_table)

# %%
for i in range(len(dfs[0].columns)):
    print(dfs[0].columns[i], dist_corr_table.PNN[i])
# %%

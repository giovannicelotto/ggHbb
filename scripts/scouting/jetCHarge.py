# %%
import uproot
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

# %%
dataFileNames = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/Data_10.parquet"
signalFileNames = glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/*.parquet")

# %%
dfD = pd.read_parquet(dataFileNames)
dfS = pd.read_parquet(signalFileNames)

# %%
fig, ax =plt.subplots(1, 1)
ax.hist(dfS.jet1_chargeK1+dfS.jet2_chargeK1, bins=np.linspace(-2, 2, 21), histtype='step', label='Signal', density=True)
ax.hist(dfD.jet1_chargeK1+dfD.jet2_chargeK1, bins=np.linspace(-2, 2, 21), histtype='step', label='Data', density=True)
# %%
fig, ax =plt.subplots(1, 1)
ax.hist2d(dfS.jet1_chargeUnweighted+dfS.jet2_chargeUnweighted,dfS.jeDtpwfd6dT3
t1_chargeKp1+dfS.jet2_chargeKp1, bins=(np.linspace(-2, 2, 21), np.linspace(-2, 2, 21)), label='Signal', density=True)
fig, ax =plt.subplots(1, 1)
ax.hist2d(dfD.jet1_chargeUnweighted+dfD.jet2_chargeUnweighted,dfD.jet1_chargeKp1+dfD.jet2_chargeKp1, bins=(np.linspace(-2, 2, 21), np.linspace(-2, 2, 21)), label='Signal', density=True)

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
# %%
outFolder = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched"
fileNames = glob.glob(outFolder+"/*.parquet")
df = pd.read_parquet(fileNames)
# %%
df = df[df.correctChoice==True]

# %%
df['maxChargeP3'] = np.where(df['jet1_chargeP3'] > df['jet2_chargeP3'], df['jet1_chargeP3'], df['jet2_chargeP3'])
df['minChargeP3'] = np.where(df['jet1_chargeP3'] > df['jet2_chargeP3'], df['jet2_chargeP3'], df['jet1_chargeP3'])
fig, ax = plt.subplots(1, 1)
ax.hist(df.maxChargeP3, bins=np.linspace(-3, 3, 20), histtype=u'step')
ax.hist(df.minChargeP3, bins=np.linspace(-3, 3, 20), histtype=u'step')
ax.text(s="Mean : %.2f"%(df.maxChargeP3.mean()), x = 0.9, y = 0.6, transform=ax.transAxes, ha='right')
ax.text(s="Mean : %.2f"%(df.minChargeP3.mean()), x = 0.9, y = 0.55, transform=ax.transAxes, ha='right')
# %%
# %%
df['maxChargeP5'] = np.where(df['jet1_chargeP5'] > df['jet2_chargeP5'], df['jet1_chargeP5'], df['jet2_chargeP5'])
df['minChargeP5'] = np.where(df['jet1_chargeP5'] > df['jet2_chargeP5'], df['jet2_chargeP5'], df['jet1_chargeP5'])
fig, ax = plt.subplots(1, 1)
ax.hist(df.maxChargeP5, bins=np.linspace(-2, 2, 20), histtype=u'step')
ax.hist(df.minChargeP5, bins=np.linspace(-2, 2, 20), histtype=u'step')
ax.text(s="Mean : %.2f"%(df.maxChargeP5.mean()), x = 0.9, y = 0.6, transform=ax.transAxes, ha='right')
ax.text(s="Mean : %.2f"%(df.minChargeP5.mean()), x = 0.9, y = 0.55, transform=ax.transAxes, ha='right')
# %%
df['maxChargeP1'] = np.where(df['jet1_chargeP1'] > df['jet2_chargeP1'], df['jet1_chargeP1'], df['jet2_chargeP1'])
df['minChargeP1'] = np.where(df['jet1_chargeP1'] > df['jet2_chargeP1'], df['jet2_chargeP1'], df['jet1_chargeP1'])
fig, ax = plt.subplots(1, 1)
ax.hist(df.maxChargeP1, bins=np.linspace(-4, 4, 20), histtype=u'step')
ax.hist(df.minChargeP1, bins=np.linspace(-4, 4, 20), histtype=u'step')
ax.text(s="Mean : %.2f"%(df.maxChargeP1.mean()), x = 0.9, y = 0.6, transform=ax.transAxes, ha='right')
ax.text(s="Mean : %.2f"%(df.minChargeP1.mean()), x = 0.9, y = 0.55, transform=ax.transAxes, ha='right')
# %%
df['maxChargeP7'] = np.where(df['jet1_chargeP7'] > df['jet2_chargeP7'], df['jet1_chargeP7'], df['jet2_chargeP7'])
df['minChargeP7'] = np.where(df['jet1_chargeP7'] > df['jet2_chargeP7'], df['jet2_chargeP7'], df['jet1_chargeP7'])
fig, ax = plt.subplots(1, 1)
ax.hist(df.maxChargeP7, bins=np.linspace(-2, 2, 20), histtype=u'step')
ax.hist(df.minChargeP7, bins=np.linspace(-2, 2, 20), histtype=u'step')
ax.text(s="Mean : %.2f"%(df.maxChargeP7.mean()), x = 0.9, y = 0.6, transform=ax.transAxes, ha='right')
ax.text(s="Mean : %.2f"%(df.minChargeP7.mean()), x = 0.9, y = 0.55, transform=ax.transAxes, ha='right')
# %%
df['maxCharge'] = np.where(df['jet1_charge'] > df['jet2_charge'], df['jet1_charge'], df['jet2_charge'])
df['minCharge'] = np.where(df['jet1_charge'] > df['jet2_charge'], df['jet2_charge'], df['jet1_charge'])
fig, ax = plt.subplots(1, 1)
ax.hist(df.maxCharge, bins=np.linspace(-1, 1, 20), histtype=u'step')
ax.hist(df.minCharge, bins=np.linspace(-1, 1, 20), histtype=u'step')
ax.text(s="Mean : %.2f"%(df.maxCharge.mean()), x = 0.9, y = 0.6, transform=ax.transAxes, ha='right')
ax.text(s="Mean : %.2f"%(df.minCharge.mean()), x = 0.9, y = 0.55, transform=ax.transAxes, ha='right')
# %%
upMean = np.array([df.maxChargeP1.mean(), df.maxChargeP3.mean(), df.maxChargeP5.mean(), df.maxChargeP7.mean(), df.maxCharge.mean()])
downMean = np.array([df.minChargeP1.mean(), df.minChargeP3.mean(), df.minChargeP5.mean(), df.minChargeP7.mean(), df.minCharge.mean()])
upStd = np.array([df.maxChargeP1.std(), df.maxChargeP3.std(), df.maxChargeP5.std(), df.maxChargeP7.std(), df.maxCharge.std()])
downStd = np.array([df.minChargeP1.std(), df.minChargeP3.std(), df.minChargeP5.std(), df.minChargeP7.std(), df.minCharge.std()])
# %%
fig, ax =plt.subplots(1, 1)
ax.errorbar(x=[0.1, 0.3, 0.5, 0.7, 1], y = upMean, yerr=upStd, marker='o', linestyle='none', label='Jet with highest charge')
ax.errorbar(x=[0.11, 0.31, 0.51, 0.71, 1.01], y = downMean, yerr=downStd, marker='o', linestyle='none', label='Jet with Lowest charge')
nev=4
ax.errorbar(x=[0.1, 0.3, 0.5, 0.7, 1.], y = [df.maxChargeP1.iloc[nev], df.maxChargeP3.iloc[nev], df.maxChargeP5.iloc[nev], df.maxChargeP7.iloc[nev], df.maxCharge.iloc[nev]], marker='x', linestyle='none', label='Ev0')
ax.errorbar(x=[0.1, 0.3, 0.5, 0.7, 1.], y = [df.minChargeP1.iloc[nev], df.minChargeP3.iloc[nev], df.minChargeP5.iloc[nev], df.minChargeP7.iloc[nev], df.minCharge.iloc[nev]], marker='x', linestyle='none', label='Ev1')
#nev=5
#ax.errorbar(x=[0.1, 0.3, 0.5, 0.7, 1.], y = [df.maxChargeP1.iloc[nev], df.maxChargeP3.iloc[nev], df.maxChargeP5.iloc[nev], df.maxChargeP7.iloc[nev], df.maxCharge.iloc[nev]], marker='x', linestyle='none', label='Ev2')
ax.set_xlabel(r"$k$")
ax.set_ylabel("Mean of Charge")
ax.legend()
# %%

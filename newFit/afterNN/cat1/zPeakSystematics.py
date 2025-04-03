# %%
import numpy as np
import json, sys
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from helpers.allFunctions import *
from helpers.fitWithSystematics import *
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from functions import  getDfProcesses_v2

# %%
MCList = [1, 3, 4, 19,20, 21, 22]
x1, x2 = 40, 300
set_x_bounds(x1, x2)
modelName = "Mar21_1_0p0"
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
dfProcesses = getDfProcesses_v2()[0].iloc[MCList]
outFolder = "/t3home/gcelotto/ggHbb/newFit/afterNN/cat1"
cuts_dict = {
        "dijet_pt": (100, 160), 
        "PNN": (0.575, None),      
}


# %%
fitFunction='zPeak_dscb'
fit_systematics = FitWithSystematics(modelName, path, dfProcesses, x1, x2, outFolder, fitFunction)
# %%
dfsMC = fit_systematics.load_data()
dfsMC = fit_systematics.apply_cuts(dfsMC, cuts_dict)
bins = np.linspace(x1, x2, 50)
x = (bins[1:] + bins[:-1]) / 2
cTot = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)
for idx, df in enumerate(dfsMC):
    c=np.histogram(df.dijet_mass, bins=bins, weights=df.weight)[0]
    cerr=np.histogram(df.dijet_mass, bins=bins, weights=(df.weight)**2)[0]
    err = err + cerr

    cTot=cTot+c
err = np.sqrt(err)
fitregion = ((x > x1) & (x < x2))
# %%
m = fit_systematics.fit_model(x, cTot, err, fitregion)
# %%
fit_systematics.save_parameters("nominal", m)
# %%
fit_systematics.plot_results(x, cTot, err, m, fitregion, bins, outFolder)
# %%
variations = ['jet1_btag_up', 'jet1_btag_down']
fit_systematics.apply_variations(variations, dfsMC, bins, fitregion)
fit_systematics.plot_variations_results(x, cTot, err, bins, fitregion, outFolder)
fit_systematics.save_results()
# %%

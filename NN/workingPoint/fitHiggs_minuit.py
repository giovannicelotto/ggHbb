# %%
#from fitFunction import DoubleSidedCrystalballFunctionPDF_normalized
import glob, re, sys
sys.path.append('/t3home/gcelotto/ggHbb/NN')
import numpy as np
import pandas as pd
from functions import loadMultiParquet, cut, getXSectionBR
from helpersForNN import preprocessMultiClass, scale, unscale
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import mplhep as hep
hep.style.use("CMS")
import math
from applyMultiClass_Hpeak import getPredictions, splitPtFunc
from iminuit import cost
from iminuit import Minuit
from numba_stats import crystalball_ex
# %%

#
# load Data
#

from loadData import loadData
dfs, YPred_H = loadData()

# %%
xmin = 70
xmax = 200

workingPoint = (YPred_H[:,1]>0.34) & (YPred_H[:,0]<0.22)

fig,  ax = plt.subplots(1, 1)
bins=np.linspace(xmin, xmax, 100)
counts = np.histogram(dfs[0].dijet_mass[workingPoint], bins=bins, weights=dfs[0].sf[workingPoint])[0]
ax.hist(bins[:-1], bins=bins, weights=counts)
ax.set_xlim(xmin, xmax)

# %%

costFunc = cost.UnbinnedNLL(dfs[0].dijet_mass[workingPoint], crystalball_ex.pdf)
truth = (1.2, 28, 19, 1.6, 8.01, 14, 124)
m = Minuit(costFunc, *truth)
m.limits["beta_left", "beta_right"] = (0, 3)
m.limits["m_left"] = (0, 100)
m.limits["m_right"] = (0, 20)
m.limits["scale_left", "scale_right"] = (10, 30)
m.limits["loc"] = (115, 130)
#desired_edm = 1e-3  # This is your goal
#m.tol = desired_edm  

m.migrad(ncall=10000)  # finds minimum of least_squares function
#m.hesse()   # accurately computes uncertainties

# %%
print(m.params) # parameter info (using str(m.params))
print(m.covariance)


# %%
m.fval/(len(dfs[0][workingPoint])-7)
# %%

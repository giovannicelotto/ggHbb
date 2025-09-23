# %%
import numpy as np
import sys
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from helpers.allFunctions import *
from helpers.fitWithSystematics import *
import mplhep as hep
hep.style.use("CMS")
from functions import getDfProcesses_v2
import yaml
import argparse
import os
# %%

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="2", help='Config File')
parser.add_argument('-p', '--particle', type=str, default='Z', help='H or Z')
if hasattr(sys, 'ps1') or not sys.argv[1:]:
    # Interactive mode (REPL, Jupyter) OR no args provided → use defaults
    args = parser.parse_args([])
else:
    # Normal CLI usage
    args = parser.parse_args()



with open("/t3home/gcelotto/ggHbb/WSFit/Configs/cat"+str(args.config)+".yml", 'r') as f:
        config = yaml.safe_load(f)

modelName = config['modelName']
outFolder = config['outFolder']
if not os.path.exists(outFolder):
    c = input("OutFolder not found. Do you want to create it [y/n]")
    if c=="y":
        os.makedirs(outFolder)
        os.makedirs(outFolder+"/plots")
    elif c =="n":
        sys.exit("Closing")
    else:
        sys.exit("Not recognized")
    

cuts_string = config['cuts_string']


if args.particle=='H':
    MCList = config['H_MCList']
    MCList_sD = config['H_MCList_sD']
    MCList_sU = config['H_MCList_sU']
    x1, x2 = config['H_x_bounds']
    fitFunction= config['H_fitFunction']
    nbins = config["H_nbins"]
    params = config["H_params"]
    paramsLimits  = config["H_paramsLimits"]
elif args.particle=='Z':
    MCList = config['Z_MCList']
    MCList_sD = config['Z_MCList_sD']
    MCList_sU = config['Z_MCList_sU']
    x1, x2 = config['Z_x_bounds']
    fitFunction= config['Z_fitFunction']
    nbins = config["Z_nbins"]
    params = config["Z_params"]
    paramsLimits  = config["Z_paramsLimits"]
else:
    assert False, "args.particle must be either H or Z. Value not valid"


path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
dfProcesses = getDfProcesses_v2()[0].iloc[MCList]
dfProcesses_sD = getDfProcesses_v2()[0].iloc[MCList_sD]
dfProcesses_sU = getDfProcesses_v2()[0].iloc[MCList_sU]

set_x_bounds(x1, x2)


# %%

fit_systematics = FitWithSystematics(modelName, path, dfProcesses, x1, x2, outFolder, fitFunction, dfProcesses_sD=dfProcesses_sD, dfProcesses_sU=dfProcesses_sU, particle=args.particle)
# %%
dfsMC = fit_systematics.load_data()
print("These are all the columns:")
print(dfsMC[0].columns)
print([len(dfsMC[i]) for i in range(len(dfsMC))])
dfsMC = fit_systematics.apply_cuts(dfsMC, cuts_string)
bins = np.linspace(x1, x2, nbins)
x = (bins[1:] + bins[:-1]) / 2
cTot = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)
for idx, df in enumerate(dfsMC):
    print(df.dijet_mass.isna().sum(), df.weight.isna().sum())
    print("Infs in weight:", np.isinf(df['weight']).sum())
    print("Infs in genWeight:", np.isinf(df['genWeight']).sum())
    #print("Infs in NLO_kfactor:", np.isinf(df['NLO_kfactor']).sum())
    print("Infs in btag_central:", np.isinf(df['btag_central']).sum())
    print("Infs in sf:", np.isinf(df['sf']).sum())
    c=np.histogram(df.dijet_mass, bins=bins, weights=df.weight)[0]
    print(df.dijet_mass.max(), df.dijet_mass.min(),  df.dijet_mass.mean(), len(df.dijet_mass))
    cerr=np.histogram(df.dijet_mass, bins=bins, weights=(df.weight)**2)[0]
    err = err + cerr

    cTot=cTot+c
err = np.sqrt(err)
fitregion = ((x > x1) & (x < x2))

# %%
m = fit_systematics.fit_model(x, cTot, err, fitregion, params, paramsLimits)
print(m.values)
# %%
fit_systematics.save_parameters("nominal", m)
# %%

fit_systematics.plot_results(x, cTot, err, m, fitregion, bins, outFolder)
# %%
variations = [
    'btag_up', 'btag_down',
    ]
fit_systematics.apply_variations(variations, dfsMC, bins, fitregion, params, paramsLimits, cuts_string)
fit_systematics.plot_variations_results(x, cTot, err, bins, fitregion, outFolder)
fit_systematics.save_results()
# %%

# %%
from config import load_config
from io_ttbar import load_all_data
from processing import prepare_mc, apply_cuts
from histograms import make_histograms
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
hep.style.use("CMS")
from plotStacked import plot_all_variables
from build_systematics_ttbar import *
# %%
category=0
print(f"Running for category {category}...")
config = load_config(category)
# %%
#config['df_folder']="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Jan21_3_50p0_Apr"
print("Loading data...")
dfMC, dfData, lumi = load_all_data(config)
print("Data loaded. Preparing MC...")
dfMC = prepare_mc(dfMC, lumi)
print("Applying cuts...")
# %%

dfData = apply_cuts_single(dfData, config)
# Cuts for MC are applied after systematic uncertainties variations
# Dont apply here!
#dfMC = apply_cuts_single(dfMC, config)


# %%
# NEW STEP
#config['systematics']=['nominal']
mc_variations = build_systematics(dfMC, config)

# %%

make_histograms(mc_variations, dfData, config, category, n_bins=4)
# %%
plot_all_variables(dfData, mc_variations['nominal'], lumi, config, folder=f"/t3home/gcelotto/ggHbb/tt_CR/plots/controlPlots/{category}")
# %%

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
print("Loading data...")
dfMC, dfData, lumi = load_all_data(config)
print("Data loaded. Preparing MC...")
dfMC = prepare_mc(dfMC, lumi)
print("Applying cuts...")
dfData = apply_cuts_single(dfData, config)


# %%
# NEW STEP
mc_variations = build_systematics(dfMC, config)

# %%

make_histograms(mc_variations, dfData, config, category)
# %%
plot_all_variables(dfData, mc_variations['nominal'], lumi, config)
# %%

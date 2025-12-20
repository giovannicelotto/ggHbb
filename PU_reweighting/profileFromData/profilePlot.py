# %%
import matplotlib.pyplot as plt
import uproot
import ROOT
import mplhep as hep
import numpy as np
import json
import pandas as pd
import mplhep as hep
hep.style.use("CMS")
# %%
pathToData="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/profileBParking"
dataFile = pathToData+"/pileup_data_2018.root"
data= uproot.open(dataFile)
hist=data['pileup_2018total']
bin_contents_data = hist.values()
bin_edges = hist.axis().edges()
assert len(bin_contents_data)==len(bin_edges)-1
# %%
import importlib.util
import sys
pathMC ="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/profileBParking/mix_2018_25ns_UltraLegacy_PoissonOOTPU_cfi.py"
spec = importlib.util.spec_from_file_location("myPU", pathMC)
module = importlib.util.module_from_spec(spec)
sys.modules["myPU"] = module
spec.loader.exec_module(module)
xMC = list(module.mix.input.nbPileupEvents.probFunctionVariable)
yMC = list(module.mix.input.nbPileupEvents.probValue)



# %%
# For MC taken from the campaign
# /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/profileBParking/mix_2018_25ns_UltraLegacy_PoissonOOTPU_cfi.py


assert len(xMC)==len(yMC)
# %%
# Plot
# yMC is already normalized to have area = 1
# Data are normalized in the same way by hand
# Final check is done
fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                       gridspec_kw={'height_ratios': [4, 1]})
integral = sum(bin_contents_data*np.diff(bin_edges))
normalized_bin_contents_data = bin_contents_data / integral
hep.histplot((normalized_bin_contents_data, bin_edges), ax=ax[0], label='Data')
ax[0].step(xMC, yMC, label='MC UL2018')
ax[0].set_xlim(0, 100)
ax[0].legend()

ax[0].text(x=0.05, y=0.6, s="Integral Data : %.3f"%sum(bin_contents_data / integral), transform=ax[0].transAxes, ha='left')
ax[0].text(x=0.05, y=0.55, s="Integral MC : %.3f"%sum(yMC), transform=ax[0].transAxes, ha='left')
ax[1].set_xlabel("Pileup")
ax[0].set_ylabel("Normalized Units")
#ax[0].set_yscale('log')
epsilon = 1e-12
ratio = np.where(np.array(yMC)!=0,
                 bin_contents_data[:99]/integral/(np.array(yMC)+epsilon),
                                                  0)
#ratio = bin_contents_data[:99]/integral/(np.array(yMC)+epsilon)
ax[1].plot(ratio, marker='o', color='black', linestyle='none')
ax[1].set_ylabel("Data/MC")
ax[1].set_ylim(0, 2)
fig.savefig("/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_profile.png")

# %%
mymap = dict(zip(xMC, ratio))
mymap = {int(k): v for k, v in mymap.items()}
with open('/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_PVtoPUSF.json', 'w') as file:
    json.dump(mymap, file, indent=4)


# %%

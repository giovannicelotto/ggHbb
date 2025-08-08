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
# For MC taken from the campaign
# /pnfs/psi.ch/cms/trivcat/store/user/gcelotto/profileBParking/mix_2018_25ns_UltraLegacy_PoissonOOTPU_cfi.py
xMC = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95, 96, 97, 98
    )

yMC = (
    8.89374611122e-07, 1.1777062868e-05, 3.99725585118e-05, 0.000129888015252, 0.000265224848687,
    0.000313088635109, 0.000353781668514, 0.000508787237162, 0.000873670065767, 0.00147166880932,
    0.00228230649018, 0.00330375581273, 0.00466047608406, 0.00624959203029, 0.00810375867901,
    0.010306521821, 0.0129512453978, 0.0160303925502, 0.0192913204592, 0.0223108613632,
    0.0249798930986, 0.0273973789867, 0.0294402350483, 0.031029854302, 0.0324583524255,
    0.0338264469857, 0.0351267479019, 0.0360320204259, 0.0367489568401, 0.0374133183052,
    0.0380352633799, 0.0386200967002, 0.039124376968, 0.0394201612616, 0.0394673457109,
    0.0391705388069, 0.0384758587461, 0.0372984548399, 0.0356497876549, 0.0334655175178,
    0.030823567063, 0.0278340752408, 0.0246009685048, 0.0212676009273, 0.0180250593982,
    0.0149129830776, 0.0120582333486, 0.00953400069415, 0.00738546929512, 0.00563442079939,
    0.00422052915668, 0.00312446316347, 0.00228717533955, 0.00164064894334, 0.00118425084792,
    0.000847785826565, 0.000603466454784, 0.000419347268964, 0.000291768785963, 0.000199761337863,
    0.000136624574661, 9.46855200945e-05, 6.80243180179e-05, 4.94806013765e-05, 3.53122628249e-05,
    2.556765786e-05, 1.75845711623e-05, 1.23828210848e-05, 9.31669724108e-06, 6.0713272037e-06,
    3.95387384933e-06, 2.02760874107e-06, 1.22535149516e-06, 9.79612472109e-07, 7.61730246474e-07,
    4.2748847738e-07, 2.41170461205e-07, 1.38701083552e-07, 3.37678010922e-08, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0
    )

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

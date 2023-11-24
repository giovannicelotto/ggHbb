import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

old = np.load("/t3home/gcelotto/ggHbb/outputs/ggHbb_bScoreBased4_01.npy")
new = np.load("/t3home/gcelotto/ggHbb/outputs/ggHbb_bScoreBased4_11.npy")
std_old = np.std(old[old[:,21]<200,21])
std_new = np.std(new[new[:,21]<200,21])
print(old.shape, new.shape)
fig, ax = plt.subplots(1, 1)
ax.hist(old[:,21], bins=np.linspace( 0, 500, 100), color='blue', histtype=u'step', label='old (std:%.1f)'%std_old)
ax.hist(new[:,21], bins=np.linspace( 0, 500, 100), color='red', histtype=u'step', label='new (std:%.1f)'%std_new)
ax.set_ylim(0, ax.get_ylim()[1])
ax.vlines(x=125.09, ymin=0, ymax=ax.get_ylim()[1], label='125.09', color='black', linestyles='---')
ax.legend()
fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/compareTraining.pdf", bbox_inches='tight')

# %%
import numpy as np
l = np.load("/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/Aug28_3_20p01/featuresForTraining.npy")

# %%
for i in range(len(l)):
    if "dimuon" not in l[i] :
        l[i] = l[i].replace('muon_', 'jet1_muon_')
# %%
np.save("/t3home/gcelotto/ggHbb/PNN/results_mjjDisco/Aug28_3_20p01/model/featuresForTraining.npy", l)
# %%

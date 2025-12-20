# %%
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from functions import getDfProcesses_v2
import yaml
# %%
openingFile = open("/t3home/gcelotto/ggHbb/documentation/plotScripts/nanoToFlat_cfg.yaml", "r")
cfg = yaml.load(openingFile, Loader=yaml.FullLoader)

grouped_pNs=cfg['processes_grouped']
pNs = [pN for group in grouped_pNs.values() for pN in group]
# %%


# %%
dfProcesses = getDfProcesses_v2()[0]

group_steps_loaded = np.load("/t3home/gcelotto/ggHbb/documentation/outputScripts_forPlotting/cutflow_mini_to_regions.npy", allow_pickle=True).item()
# %%
processes = list(group_steps_loaded.keys())

# take labels from the first process (same cutflow for all)
step_labels = list(next(iter(group_steps_loaded.values())).keys())[:-3]
x = np.arange(len(step_labels))

n_proc = len(processes)
total_width = 0.8
bar_width = total_width / n_proc

fig, ax = plt.subplots(1, 1, figsize=(15, 6))

for i, pr in enumerate(processes):
    ys = list(group_steps_loaded[pr].values())
    offset = (i - n_proc / 2) * bar_width + bar_width / 2
    ax.bar(x + offset, ys, width=bar_width, label=pr)
#ax.vlines(x=.5, ymin=0,ymax=10**11)
ax.set_xticks(x)
ax.set_xticklabels(step_labels, rotation=30, ha="right")
ax.set_yscale("log")
ax.grid(True)
ax.set_ylabel("Yields")
#ax.set_title("Cutflow yields by process")
ax.legend(ncol=2, fontsize='x-small')
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/cutflow_mini_to_regions.png", bbox_inches='tight')
# %%

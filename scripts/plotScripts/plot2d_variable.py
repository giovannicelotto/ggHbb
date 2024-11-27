# %%
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplhep as hep
hep.style.use("CMS")
import json
class DictionaryLoader:
    _instance = None
    _mapping_dict = None

    def __new__(cls, file_path):
        if cls._instance is None:
            cls._instance = super(DictionaryLoader, cls).__new__(cls)
            cls._load_dict(file_path)
        return cls._instance

    @classmethod
    def _load_dict(cls, file_path):
        with open(file_path, 'r') as file:
            # Use json.load and convert keys to integers
            mapping_data = json.load(file)
            cls._mapping_dict = {int(k): v for k, v in mapping_data.items()}
            # Determine the last x value and the corresponding y value
            cls._last_x_value = max(cls._mapping_dict.keys())
            cls._last_y_value = cls._mapping_dict[cls._last_x_value]

    @classmethod
    def get_dict(cls):
        return cls._mapping_dict
# %%

flatPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
flatFileNames = glob.glob(flatPath+"/QCD_Pt20To30/**/*.parquet", recursive=True)[:100]
columns=['muon_pt', 'PV_npvs']
x_bins, y_bins = np.linspace(20, 40, 20), np.linspace(0, 1, 20)
df = pd.read_parquet(flatFileNames, columns=columns)
dictionary_loader = DictionaryLoader('/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_PVtoPUSF.json')
PU_map = dictionary_loader.get_dict()
df['PU_SF'] = df['PV_npvs'].map(PU_map)
columns.append('PU_SF')
columns.remove('PV_npvs')
#flatFileNames = glob.glob(flatPath+"/Data1A/**/*.parquet", recursive=True)[:10]
#df_data = pd.read_parquet(flatFileNames, columns=['jet1_btagDeepFlavB', 'jet1_pt','dijet_twist'])


fig, ax_main = plt.subplots(figsize=(8, 8))
divider = make_axes_locatable(ax_main)
ax_top = divider.append_axes("top", 1.2, pad=0.2, sharex=ax_main)
ax_right = divider.append_axes("right", 1.2, pad=0.2, sharey=ax_main)

# Plot the 2D histogram in the main axes
hist, x_edges, y_edges = np.histogram2d(x=df[columns[0]], y=df[columns[1]], bins=[x_bins, y_bins])
ax_main.imshow(hist.T, origin='lower', extent=(x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()), aspect='auto', cmap='coolwarm')
ax_main.set_xlabel(columns[0])
ax_main.set_ylabel(columns[1])

# Plot the marginalized histogram on top
ax_top.hist(df[columns[0]], bins=x_bins, color='lightblue', edgecolor='black')
ax_top.set_xlim(ax_main.get_xlim())
ax_top.set_yticks([])
ax_top.xaxis.tick_top()

# Plot the marginalized histogram on the right
ax_right.hist(df[columns[1]], bins=y_bins, color='lightblue', edgecolor='black', orientation='horizontal')#lightcoral
ax_right.set_ylim(ax_main.get_ylim())
ax_right.set_xticks([])
ax_right.yaxis.tick_right()


#outName = "/t3home/gcelotto/ggHbb/outputs/plots/hist2d_jet1btag_dijettwist.png"
#fig.savefig(outName, bbox_inches='tight')
#print("Saving in ", outName)

# %%

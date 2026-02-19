# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
# %%
X = pd.read_parquet("/t3home/gcelotto/ggHbb/PNN/input/data_pt3_1D/Xtrain.parquet")
# read the config in yaml:
configPath = "/t3home/gcelotto/ggHbb/PNN/config/featuresToRead.yaml"
import yaml
with open(configPath, "r") as f:
    config = yaml.safe_load(f)
# %%


corr = X[config["featuresForTraining"]].corr()
labels = corr.columns

fig, ax = plt.subplots(1, 1)

cax = ax.matshow(corr)
fig.colorbar(cax)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)
ax.tick_params(axis='x', which='major', labelsize=8)
ax.tick_params(axis='y', which='major', labelsize=8)
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/PNN/correlation_matrix.pdf", bbox_inches='tight')
# %%
import numpy as np

corr = X[config["featuresForTraining"]].corr()

# Maschera per tenere solo il triangolo superiore, escludendo la diagonale
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)

corr_pairs = (
    corr.where(mask)
        .stack()
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
)

# Ordina per correlazione assoluta (dalla più forte)
corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()
corr_pairs = corr_pairs.sort_values("abs_correlation", ascending=False)

corr_pairs.head(20)

# %%
import matplotlib.pyplot as plt
import numpy as np

top_n = 20
top_corr = corr_pairs.head(top_n)

labels = top_corr["feature_1"] + " vs " + top_corr["feature_2"]
values = top_corr["abs_correlation"]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.barh(labels, values)
for bar in ax.patches:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    ax.text(
        width,
        y,
        f"{width:.3f}",
        va="center",
        ha="left",
        fontsize=10
    )
ax.invert_yaxis()  # la più alta in cima

ax.set_xlabel("Absolute correlation")
ax.set_ylabel("Feature pairs")
#ax.set_title("Top correlated feature pairs")

plt.tight_layout()
plt.show()
fig.savefig("/t3home/gcelotto/ggHbb/documentation/plots/PNN/top_correlated_features.pdf", bbox_inches='tight')  

# %%

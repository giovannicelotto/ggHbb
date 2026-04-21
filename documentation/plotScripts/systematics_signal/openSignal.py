# %%
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import yaml
path ="/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/dataframes_NN/Jan21_3_50p0/df_GluGluHToBBMINLO_Jan21_3_50p0.parquet"
df = pd.read_parquet(path)
# %%
# List of columns starting with NN_Jet_sys
list_nn_sys = [col for col in df.columns if col.startswith("NN_Jet_sys")]


# %%
import numpy as np
import matplotlib.pyplot as plt

configs = [1, 7, 8]
path_config = "/t3home/gcelotto/ggHbb/WSFit/Configs/"

fig, ax = plt.subplots(2, len(configs), figsize=(5*len(configs), 8), sharex=True)

bins = np.linspace(100, 150, 11)

for idx, cfg_nmbr in enumerate(configs):
    cfg_file = path_config + "cat%d.yml" % cfg_nmbr
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # --- nominal ---
    df_nom = df.query(cfg["cuts_string"])
    weights_nom = df_nom["weight"] * 41.6

    hist_nom, _ = np.histogram(df_nom["dijet_mass"], bins=bins, weights=weights_nom)
    err_nom, _ = np.histogram(df_nom["dijet_mass"], bins=bins, weights=weights_nom**2)
    err_nom = np.sqrt(err_nom)

    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # plot nominal
    ax[0, idx].step(bin_centers, hist_nom, where="mid", label="nominal")

    # --- systematics ---
    for col in list_nn_sys:
        new_string = cfg["cuts_string"].replace("PNN", col)
        df_var = df.query(new_string)

        weights_var = df_var["weight"] * 41.6

        hist_var, _ = np.histogram(df_var["dijet_mass"], bins=bins, weights=weights_var)
        err_var, _ = np.histogram(df_var["dijet_mass"], bins=bins, weights=weights_var**2)
        err_var = np.sqrt(err_var)

        # --- ratio ---
        ratio = np.divide(hist_var, hist_nom, out=np.zeros_like(hist_var), where=hist_nom!=0)

        # error propagation
        ratio_err =  (err_var / hist_nom)
        ratio_err = np.nan_to_num(ratio_err)

        # --- plots ---
        ax[0, idx].step(bin_centers, hist_var, where="mid", alpha=0.5, color="green" if "up" in col else "red", label=col)

        ax[1, idx].errorbar(
            bin_centers, ratio, yerr=ratio_err,
            fmt="o", markersize=3, alpha=0.7, color="green" if "up" in col else "red",
        )
        ax[1, idx].set_ylim(0.5, 1.5)

    ax[0, idx].set_title(f"cat{cfg_nmbr}")
    ax[1, idx].axhline(1.0, linestyle="--")

# labels
ax[0, 0].set_ylabel("Events")
ax[1, 0].set_ylabel("Ratio")
for i in range(len(configs)):
    ax[1, i].set_xlabel("dijet_mass")
plt.tight_layout()
plt.show()
# %%




import numpy as np
import matplotlib.pyplot as plt

configs = [1, 7, 8]
path_config = "/t3home/gcelotto/ggHbb/WSFit/Configs/"
bins = np.linspace(100, 150, 11)

base_systs = sorted(set(col.replace("_up", "").replace("_down", "") for col in list_nn_sys))

for base in base_systs:

    col_up = base + "_up"
    col_down = base + "_down"

    if col_up not in list_nn_sys or col_down not in list_nn_sys:
        continue

    fig, ax = plt.subplots(2, len(configs), figsize=(5*len(configs), 8), sharex=True)

    for idx, cfg_nmbr in enumerate(configs):

        cfg_file = path_config + "cat%d.yml" % cfg_nmbr
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)

        # --- nominal ---
        df_nom = df.query(cfg["cuts_string"])
        w_nom = df_nom["weight"] * 41.6

        hist_nom, _ = np.histogram(df_nom["dijet_mass"], bins=bins, weights=w_nom)
        err_nom2, _ = np.histogram(df_nom["dijet_mass"], bins=bins, weights=w_nom**2)
        err_nom = np.sqrt(err_nom2)

        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # --- NOMINAL PLOT ---
        ax[0, idx].step(bin_centers, hist_nom, where="mid", label="nominal", color='black')

        # shaded uncertainty band
        ax[0, idx].fill_between(
            bin_centers,
            hist_nom - err_nom,
            hist_nom + err_nom,
            alpha=0.3,
            step="mid"
        )

        # --- variations (NO ERRORS) ---
        for variation, label in zip([col_up, col_down], ["up", "down"]):
            color = "green" if "up" in variation else "red"
            new_string = cfg["cuts_string"].replace("PNN", variation)
            df_var = df.query(new_string)

            w_var = df_var["weight"] * 41.6

            hist_var, _ = np.histogram(df_var["dijet_mass"], bins=bins, weights=w_var)

            # ratio
            ratio = np.divide(hist_var, hist_nom, out=np.zeros_like(hist_var), where=hist_nom!=0)

            # --- plots ---
            ax[0, idx].step(bin_centers, hist_var, where="mid", label=label, color=color)

            ax[1, idx].step(bin_centers, ratio, where="mid", label=label, color=color)

        # --- ratio nominal uncertainty band ---
        ratio_band = np.divide(err_nom, hist_nom, out=np.zeros_like(err_nom), where=hist_nom!=0)

        ax[1, idx].fill_between(
            bin_centers,
            1 - ratio_band,
            1 + ratio_band,
            alpha=0.3,
            step="mid"
        )

        ax[1, idx].axhline(1.0, linestyle="--", color='black')
        ax[1, idx].set_ylim(0.9, 1.1)

        ax[0, idx].set_title(f"cat{cfg_nmbr}")

    # labels
    ax[0, 0].set_ylabel("Events")
    ax[1, 0].set_ylabel("Ratio")

    for i in range(len(configs)):
        ax[1, i].set_xlabel("dijet_mass")

    ax[0, 0].legend()
    ax[1, 0].legend()

    plt.suptitle(base)
    plt.tight_layout()
    fig.savefig("/t3home/gcelotto/ggHbb/documentation/plotScripts/systematics_signal/plots/%s.png" % base)
# %%

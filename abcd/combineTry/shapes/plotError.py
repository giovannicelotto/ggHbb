# %%
import uproot
import matplotlib.pyplot as plt
import numpy as np

# %%
ratios = {}
bin_centers = None
for i in [1,3,5,7]:
    path = "/t3home/gcelotto/ggHbb/abcd/combineTry/shapes/counts_Apr01_1000p0_%dC_Nominal.root"%i
    f = uproot.open(path)
    qcd_hist = f["QCD"]

    values = qcd_hist.values()
    errors = qcd_hist.errors()
    edges = qcd_hist.axes[0].edges()
    centers = 0.5 * (edges[1:] + edges[:-1])

    # Salvo i bin centers una volta sola
    if bin_centers is None:
        bin_centers = centers

    # Calcolo l'errore relativo, evitando divisioni per zero
    ratio = np.divide(errors, values, out=np.zeros_like(errors), where=values != 0)
    ratios[f"{i}C"] = ratio

# --- Plot ---
plt.figure(figsize=(10,6))
for label, ratio in ratios.items():
    plt.plot(bin_centers, ratio, marker='o', label=label)

plt.xlabel("Bin center")
plt.ylabel("Relative uncertainty (error / content)")
plt.title("Relative error per bin for QCD histograms")
plt.legend(title="File")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
import uproot
import numpy as np
import matplotlib.pyplot as plt

# Input
lumis = [1.39, 7.46, 9.01, 11.92]
stats = [1, 3, 5, 7]
n_bins = None
bin_errors = []

# Leggiamo gli errori relativi per ogni file
for i in stats:
    path = f"/t3home/gcelotto/ggHbb/abcd/combineTry/shapes/counts_Apr01_1000p0_{i}C_Nominal.root"
    f = uproot.open(path)
    qcd_hist = f["QCD"]

    values = qcd_hist.values()
    errors = qcd_hist.errors()

    # Salva il numero di bin la prima volta
    if n_bins is None:
        n_bins = len(values)
        bin_errors = [[] for _ in range(n_bins)]

    # Calcola l'errore relativo per ogni bin
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = np.divide(errors, values, out=np.zeros_like(errors), where=values != 0)

    # Salviamo ogni errore relativo per bin
    for j in range(n_bins):
        bin_errors[j].append(rel_errors[j])

# --- Plot ---
plt.figure(figsize=(10, 6))

for i, err_vals in enumerate(bin_errors):
    plt.plot(lumis, err_vals, marker='o', label=f"Bin {i}")

plt.xlabel("Luminosity [fb⁻¹]")
plt.ylabel("Relative Error (error / content)")
plt.title("Relative Error per Bin vs Luminosity (QCD)")
plt.grid(True)
plt.legend(title="Bins", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.tight_layout()
plt.show()

# %%

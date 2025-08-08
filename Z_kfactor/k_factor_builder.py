# %%
import json
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
hep.style.use("CMS")


# Paths to input files
dy_json_path = "/t3home/gcelotto/ggHbb/Z_kfactor/output/dyee.json"
qq_json_path = "/t3home/gcelotto/ggHbb/Z_kfactor/output/qq.json"
kfactor_output_path = "/t3home/gcelotto/ggHbb/Z_kfactor/output/kfactor.json"

# Load DY data
with open(dy_json_path, "r") as f:
    dy_data = json.load(f)

# Load qq data
with open(qq_json_path, "r") as f:
    qq_data = json.load(f)

# Convert to numpy arrays
dy_counts = np.array(dy_data['pT_counts'])
qq_counts = np.array(qq_data['pT_counts'])

dy_errors = np.array(dy_data['pT_errors'])
qq_errors = np.array(qq_data['pT_errors'])

# Sanity check
assert dy_data['pT_bins'] == qq_data['pT_bins'], "Bin edges must match"

# Compute k-factor and its uncertainty using error propagation
with np.errstate(divide='ignore', invalid='ignore'):
    kfactor = np.divide(dy_counts, qq_counts, out=np.zeros_like(dy_counts), where=qq_counts != 0)
    kfactor_error = np.sqrt(
        (dy_errors / qq_counts)**2 + 
        (dy_counts * qq_errors / qq_counts**2)**2
    )
    kfactor_error = np.where(qq_counts == 0, 0, kfactor_error)

# Save k-factor
# Account for Overflow
#bins_to_save = dy_data['pT_bins']
#bins_to_save[-1] = 999999998
kfactor_dict = {
    'bins': dy_data['pT_bins'],
    'kfactor': kfactor.tolist(),
    'kfactor_error': kfactor_error.tolist()
}

with open(kfactor_output_path, "w") as f:
    json.dump(kfactor_dict, f, indent=4)

print(f"k-factor saved to {kfactor_output_path}")

# %%


# Compute bin centers
bins = np.array(dy_data['pT_bins'])
bins[-1]=400
bin_centers = (bins[:-1] + bins[1:]) / 2

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot k-factor with error bars
ax.errorbar(
    bin_centers,
    kfactor,
    xerr=np.diff(bins)/2,
    yerr=kfactor_error,
    fmt='o',
    capsize=3,
    label='k-factor (DY $\ell\ell$ / ZJetsToQQ)',
    color='black',
    ecolor='black',
    elinewidth=1,
    markerfacecolor='black'
)

# Set axis labels and title
ax.set_xlabel('LHE pT [GeV]')
ax.set_ylabel('k-factor')
ax.set_title('Z $k$-factor as a function of LHE pT')
ax.set_xlim(bins[0], bins[-1])
ax.grid(True)
ax.set_ylim(0, 4)
ax.set_xlim(30, bins[-1])
# Add grid and legend
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
average = np.average(kfactor[3:], weights=1/kfactor_error[3:]**2)
ax.text(x=0.9, y=0.7, s="k-Factor Mean : %.2f"%average, transform=ax.transAxes, ha='right')

# Layout adjustment and display
fig.tight_layout()
plt.show()

# %%
# %%

# %%
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use("CMS")
# %%
# Data
processes = [
    "GluGluHToBB",
    "EWKZtoQQ",
    "ZJetsQQ100-200",
    "ZJetsQQ200-400",
    "ZJetsQQ400-600",
    "ZJetsQQ600-800",
    "ZJetsQQ800-Inf",
]

# First step efficiencies
efficiencies = np.load("/t3home/gcelotto/ggHbb/Zbb_steps/eff1st.npy")
eff_2nd = np.load("/t3home/gcelotto/ggHbb/Zbb_steps/eff2nd.npy")
eff_3rd = np.load("/t3home/gcelotto/ggHbb/Zbb_steps/eff3rd.npy")


efficiencies = np.array(efficiencies)*100
eff_2nd = np.array(eff_2nd)*100
eff_3rd = np.array(eff_3rd)*100
# Set up bar widths and positions
bar_width = 0.25
index = np.arange(len(processes))

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 6))

# Plotting the bars
bar1 = ax.bar(index - bar_width, efficiencies, bar_width, label="Step 1 : Trigger & Muon in Jet & TrigSF x PuSF", color="skyblue", edgecolor="black")
bar2 = ax.bar(index, eff_2nd, bar_width, label="Step 2 : Jet p$_T$ > 20 & 40<m$_{jj}$<300", color="lightgreen", edgecolor="black")
bar3 = ax.bar(index + bar_width, eff_3rd, bar_width, label="Step 3 : SR fraction", color="salmon", edgecolor="black")

# Add labels and title
ax.set_xlabel("Processes", fontsize=14)
ax.set_ylabel("Efficiency [%]", fontsize=14)
ax.set_title("Efficiency Steps", fontsize=16)

# Customize the ticks and labels
ax.set_xticks(index)
ax.set_xticklabels(processes, rotation=45, ha="right", fontsize=12)
ax.tick_params(axis="y", labelsize=12)

# Add gridlines for better visualization
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Add legend
ax.legend()

# Add annotations (values on top of bars)
for i, (b1, b2, b3) in enumerate(zip(bar1, bar2, bar3)):
    ax.text(b1.get_x() + b1.get_width() / 2, b1.get_height(), f"{b1.get_height():.2f}%", ha="center", va="bottom", fontsize=10)
    ax.text(b2.get_x() + b2.get_width() / 2, b2.get_height(), f"{b2.get_height():.2f}%", ha="center", va="bottom", fontsize=10)
    ax.text(b3.get_x() + b3.get_width() / 2, b3.get_height(), f"{b3.get_height():.2f}%", ha="center", va="bottom", fontsize=10)

# Adjust layout to prevent clipping
fig.tight_layout()
ax.set_ylim(0, ax.get_ylim()[1]*1.4)
# Show the plot
fig.savefig("/t3home/gcelotto/ggHbb/Zbb_steps/allSteps.png")

# %%
eff_3rd
eff_Z = eff_3rd[1:]
from functions import getZXsections, getXSectionBR
xsections = getZXsections(EWK=True)
# %%
print("N(Z) Full Lumi in SR : ", np.sum(xsections * eff_Z * 41.6 * 1000))
print("N(H) Full Lumi in SR : ", np.sum(getXSectionBR() * eff_3rd[0] * 41.6 * 1000))
# %%

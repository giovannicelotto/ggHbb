# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from functions import getZXsections, getXSectionBR
# %%

xsections = [getXSectionBR()]+getZXsections(EWK=True)
processes = [
    "GluGluHToBB",
    "EWKZtoQQ",
    "ZJetsQQ100-200",
    "ZJetsQQ200-400",
    "ZJetsQQ400-600",
    "ZJetsQQ600-800",
    "ZJetsQQ800-Inf",
]
# %%
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(processes, xsections, color="skyblue", edgecolor="black")

# Add labels and title
ax.set_xlabel("Processes", fontsize=14)
ax.set_ylabel("Cross Section [pb]", fontsize=14)
ax.set_title("Cross Sections", fontsize=16)

# Value above bars
for bar, xsection in zip(bars, xsections):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of the bar)
        height,  # Y-coordinate (top of the bar)
        f"{xsection:.1f}",  # Text with one decimal place
        ha="center",  # Horizontal alignment
        va="bottom",  # Vertical alignment
        fontsize=10,  # Font size
        color="black"  # Text color
    )

# Customize ticks
ax.set_xticks(range(len(processes)))
ax.set_xticklabels(processes, rotation=45, ha="right", fontsize=12)
ax.tick_params(axis="y", labelsize=12)

# Add gridlines
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.set_yscale('log')
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.4)
fig.tight_layout()
fig.savefig("/t3home/gcelotto/ggHbb/Zbb_steps/xsections.png")

# %%

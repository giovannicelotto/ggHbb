# %%
import subprocess
import yaml
import numpy as np
from pathlib import Path
import glob
import os
# %%
import numpy as np
import itertools
N_EDGES = 3
N_GRID = 10
edge_grid = np.linspace(0.7, 0.95, N_GRID)  # example scan region

classes_list = []

for e1, e2, e3 in itertools.combinations(edge_grid, N_EDGES): 
    if e1 < e2:
        classes = [
            (e1, e2),
            (e2, e3),
            (e3, 1.0),
        ]
        classes_list.append(classes)
print(f"Generated {len(classes_list)} class configurations for scanning.")
for cl in classes_list:
    print(cl)
# %%
import subprocess
from pathlib import Path
import numpy as np

PLACEHOLDER_DIR = Path("/t3home/gcelotto/ggHbb/WSFit/Configs/placeholders")
LIVE_CONFIG_DIR = Path("/t3home/gcelotto/ggHbb/WSFit/Configs")
DATACARD_DIR = Path("/t3home/gcelotto/ggHbb/WSFit/datacards")

# -----------------------------
# Function to write live YAML
# -----------------------------
def prepare_cat_config(cat_idx, pnn_low, pnn_high):
    placeholder_path = PLACEHOLDER_DIR / f"cat{cat_idx}.yml"
    live_path = LIVE_CONFIG_DIR / f"cat{cat_idx}.yml"
    text = placeholder_path.read_text()
    text = text.format(PNN_LOW=pnn_low, PNN_HIGH=pnn_high)
    live_path.write_text(text)
    return live_path

# -----------------------------
# Function to extract r errors
# -----------------------------
def extract_r_errors_from_fit_output(fit_output):
    """
    Parse the FitDiagnostics stdout (string) to get asymmetric errors on r
    """
    for line in fit_output.splitlines():
        if line.strip().startswith("Best fit r:"):
            parts = line.split()
            r_best = float(parts[3])
            err_down, err_up = parts[4].split("/")
            err_down = float(err_down.replace("-", ""))
            err_up = float(err_up.replace("+", ""))
            return r_best, err_down, err_up
    return np.nan, np.nan, np.nan

# -----------------------------
# Main scan loop
# -----------------------------
results = []
WS_DIR = "/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched"

# Remove all previous workspaces
for f in glob.glob(f"{WS_DIR}/*.root"):
    os.remove(f)
# classes_list example: [[(0.7,e1),(e1,e2),(e2,1.0)], ...]
for pnn_classes in classes_list:
    for f in glob.glob(f"{WS_DIR}/*.root"):
        os.remove(f)

    # 1️⃣ Prepare live YAML configs
    for cat_idx, (pnn_low, pnn_high) in enumerate(pnn_classes):
        prepare_cat_config(cat_idx, pnn_low, pnn_high)

    # 2️⃣ Run fitZ.sh for all categories
    for cat in range(3):
        subprocess.run([
            "bash",
            "/t3home/gcelotto/ggHbb/WSFit/allSteps/fitZ.sh",
            "-c", str(cat),
            "-i", "0",
            "-z", "1"
        ], check=True)
#
    # 3️⃣ Combine datacards
    combined_txt = DATACARD_DIR / "combined_out.txt"
    #subprocess.run([
    #    "cd",
    #    "ggHbb/CMSSW_14_1_0_pre4/src/"])
    #subprocess.run([
    #    "cmsenv"])
    cmd = [
        "bash",
        "/t3home/gcelotto/ggHbb/WSFit/datacards/runCombined.sh"
    ]

    # 4️⃣ Run FitDiagnostics
    #cmd = [
    #    "combine",
    #    "-M", "FitDiagnostics",
    #    "-d", str(combined_txt),
    #    "-t", "-1",
    #    "--expectSignal", "1",
    #    "--X-rtd", "MINIMIZER_freezeDisassociatedParams",
    #    "--setParameterRange", "r=-30,30"
    #]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = result.stdout

    # 5️⃣ Extract signal-strength errors
    r_best, err_down, err_up = extract_r_errors_from_fit_output(stdout)
    results.append({
        "classes": pnn_classes,
        "r_best": r_best,
        "err_down": err_down,
        "err_up": err_up
    })

# -----------------------------
# Results
# -----------------------------
for res in results:
    print(res)

# %%

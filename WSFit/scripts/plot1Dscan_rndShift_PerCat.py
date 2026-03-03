# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import uproot
import time
import argparse
import os
import yaml
import numpy as np
# %%
parser = argparse.ArgumentParser(description="Enrich multipdf workspace with extra PDF.")
parser.add_argument("--idx", type=int, help="Index of the workspace", default=0)
args = parser.parse_args()

idx = args.idx
np.random.seed(728)
offset=np.random.uniform(10,100)
multiplier=np.random.uniform(10,100)
yaml_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws{idx}_pdfnames.yaml"
with open(yaml_path, "r") as f:
    pdfLabels = yaml.safe_load(f)['pdf_names']
def beauty_pdfLabels(pdfLabels_raw):
    beauty_labels = []
    for label in pdfLabels_raw:
        beauty_labels.append(label.replace("env_pdf_", "").replace("_cat%d"%idx, "").replace("_", "").replace("stein", "").replace("nential", ""))
    return beauty_labels
pdfLabels = beauty_pdfLabels(pdfLabels)
# %%

# %%

nPDFs = len(pdfLabels)

# %%
# list PDF names inside
#pdf_list = multipdf.pdfList()
#names = [pdf_list[i].GetName() for i in range(pdf_list.getSize())]
#print(names)
# %%
fileName = "/t3home/gcelotto/ggHbb/WSFit/datacards/higgsCombinefitData_Blind_Cat%dpdfDiscrete.MultiDimFit.mH125.root"%idx
file = uproot.open(fileName)
tree = file["limit"]
branches = tree.arrays(library="np")

r_Discrete = (branches["r"]+offset) * multiplier
deltaNLL_Discrete = branches["deltaNLL"]
nll0_Discrete = branches["nll0"]
nll_Discrete = branches["nll"]

r_pdf_i = []
deltaNLL_pdf_i = []
nll0_pdf_i = []
nll_pdf_i = []
print("Number of PDFs: ", nPDFs)
for pdfIdx in range(nPDFs):
    fileName = "/t3home/gcelotto/ggHbb/WSFit/datacards/higgsCombinefitData_Blind_Cat%dpdf%d.MultiDimFit.mH125.root"%(idx, pdfIdx)
    file = uproot.open(fileName)
    tree = file["limit"]
    branches = tree.arrays(library="np")
    r_pdf_i.append((branches["r"]+offset) * multiplier)
    deltaNLL_pdf_i.append(branches["deltaNLL"])
    nll0_pdf_i.append(branches["nll0"])
    nll_pdf_i.append(branches["nll"])


# %%
fig, ax = plt.subplots(1,1 )
y_Discrete=nll0_Discrete+nll_Discrete+ deltaNLL_Discrete
ax.plot(r_Discrete[1:],y_Discrete[1:], 'o', markersize=3, label="Discrete Profiling", linestyle='-', linewidth=6, color='black', alpha=0.3)
for pdfIdx in range(nPDFs):
    y_pdf_i = nll0_pdf_i[pdfIdx]+nll_pdf_i[pdfIdx]+ deltaNLL_pdf_i[pdfIdx]
    ax.plot(r_pdf_i[pdfIdx][1:],y_pdf_i[1:], 'o', markersize=5, label=pdfLabels[pdfIdx], linestyle='-', linewidth=1)



# From now on only discrete
ax.hlines(min(y_Discrete)+1, xmin=min(r_Discrete), xmax=max(r_Discrete), colors='black',alpha=0.5, linestyles='dashed', label='1 sigma')
ax.hlines(min(y_Discrete)+4, xmin=min(r_Discrete), xmax=max(r_Discrete), colors='black',alpha=0.2, linestyles='dashed', label='2 sigma')

y_Discrete_left = y_Discrete[r_Discrete < r_Discrete[np.argmin(y_Discrete)]]
r_Discrete_left = r_Discrete[r_Discrete < r_Discrete[np.argmin(y_Discrete)]]
y_Discrete_right = y_Discrete[np.argmin(y_Discrete[1:]):]
r_Discrete_right = r_Discrete[np.argmin(y_Discrete[1:]):]
x_intersectionLeft2Sigma, x_intersectionRight2Sigma = r_Discrete_left[np.argmin(np.abs(y_Discrete_left - (min(y_Discrete_left)+4)))], r_Discrete_right[np.argmin(np.abs(y_Discrete_right - (min(y_Discrete_right)+4)))]
x_intersectionLeft1Sigma, x_intersectionRight1Sigma = r_Discrete_left[np.argmin(np.abs(y_Discrete_left - (min(y_Discrete_left)+1)))], r_Discrete_right[np.argmin(np.abs(y_Discrete_right - (min(y_Discrete_right)+1)))]
rMin = r_Discrete[np.argmin(y_Discrete)]
#ax.set_xlim(x_intersectionLeft2Sigma - 100, x_intersectionRight2Sigma + 100)
#ax.set_ylim(min(y_Discrete)-1, min(y_Discrete)+6)
ax.set_xlabel("(r + RND$_1$) x RND$_2$")
ax.set_ylabel("-2 log(L) + c")
ax.errorbar(
    x=rMin,
    y=min(y_Discrete)-0.5,
    xerr=np.array([
    [rMin - x_intersectionLeft2Sigma],     # negative error
    [x_intersectionRight2Sigma - rMin]     # positive error
]),
    yerr=0,
    fmt='o',
    color='orange',
    markersize=10,
    linewidth=4,
    label=f'2 $\sigma$'
)
ax.errorbar(
    x=rMin,
    y=min(y_Discrete)-0.5,
    xerr=np.array([
    [rMin - x_intersectionLeft1Sigma],     # negative error
    [x_intersectionRight1Sigma - rMin]     # positive error
]),
    yerr=0,
    fmt='o',
    color='green',
    markersize=10,
    linewidth=4,
    label=f'1 $\sigma$'
)





ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.text(0.85, 0.95, "r = %.2f$_{%.2f}^{+%.2f}$ "%(rMin, x_intersectionLeft1Sigma-rMin, x_intersectionRight1Sigma-rMin), transform=ax.transAxes, fontsize=18, ha='right', va='top')
if not os.path.exists("/t3home/gcelotto/ggHbb/WSFit/output/cat%d/plots/multipdf_results/"%(idx)):
    print("Here")
    os.makedirs("/t3home/gcelotto/ggHbb/WSFit/output/cat%d/plots/multipdf_results/"%(idx))
fileName = "/t3home/gcelotto/ggHbb/WSFit/output/cat%d/plots/multipdf_results/rndShift_1Dscan.png"%(idx)
fig.savefig(fileName, bbox_inches='tight')
print("Saved in ", fileName)
# %%

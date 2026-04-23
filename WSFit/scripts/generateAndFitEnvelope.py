# %%
# Plot the matrix of the toys generated with PDF Idx i and fitted with PDF idx j
# To be improved by specifying category and signal strength
import numpy as np
import matplotlib.pyplot as plt
import uproot
import mplhep as hep
hep.style.use("CMS")
from scipy.stats import norm
import os
import argparse
from iminuit import Minuit
from iminuit.cost import LeastSquares
# %%
parser = argparse.ArgumentParser(description="Enrich multipdf workspace with extra PDF.")
parser.add_argument("--cat", type=int, help="Index of the workspace", default=0)
parser.add_argument("--POI", type=str, help="POI r or rateZbb", default='r')
parser.add_argument("--signalExpected", type=int, help="Signal expected", default=0)
parser.add_argument("--npdf", type=int, help="Signal expected", default=3)
args = parser.parse_args()

cat = args.cat
expectedSignal = args.signalExpected
npdf = args.npdf

# %%
pdfLabels = {0:["Expo3", "PolExpo3"],
             #10:["Bern2", "Bern3", "Expo1", "PolExpo2"],
             #100:["Bern2","Expo1","PolExpo2"]
             }
toysGenValues = list(np.arange(npdf))
fig, ax = plt.subplots(1, len(toysGenValues), figsize=(20,6))

def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
ext=f"Signal{expectedSignal}"
for toysGenFrom in toysGenValues:
    fileName = f"/t3home/gcelotto/ggHbb/WSFit/datacards/fitDiagnosticsfitPdfEnvelope_ToysGen{toysGenFrom}_{ext}_cat{cat}.root"
    if not os.path.exists(fileName):
        print(f"[WARNING] File {fileName} does not exist, skipping")
        continue
        
    file = uproot.open(fileName)

    
    tree = file["tree_fit_sb"]
    branches = tree.arrays(library="np")
    r = branches[args.POI]
    rHiErr = branches[f"{args.POI}HiErr"]
    rLoErr = branches[f"{args.POI}LoErr"]
    fit_status = branches["fit_status"]


    r = r[(fit_status==0) | (fit_status==300)]
    rHiErr = rHiErr[(fit_status==0) | (fit_status==300)]
    rLoErr = rLoErr[(fit_status==0) | (fit_status==300)]

    if ext=="Signal0":
        observable = (r)/(0.5*(rHiErr+rLoErr))    
        label = "r/$\sigma_r$"
    else:
        observable = (r-1)/(0.5*(rHiErr+rLoErr))
        label = f"(r-{expectedSignal})/$\sigma_r$"
    bins = np.linspace(-5, 5, 30)

    #ax[toysGenFrom].hist(observable, bins=bins, histtype="stepfilled", alpha=0.5)[0]
    #c=ax[toysGenFrom].hist(observable[(observable>-3) & (observable<3)], bins=bins, histtype="stepfilled", alpha=0)[0]
    c=ax[toysGenFrom].hist(observable, bins=bins, histtype="stepfilled", alpha=0.5)[0]
    #Fit a gaussian with scipy
    centers = 0.5 * (bins[1:] + bins[:-1])
    errors = np.sqrt(c)  
    errors=np.where(errors==0, 1, errors)  # Avoid zero division
    lsq = LeastSquares(centers, c, errors, gauss)

    # --- Fit ---
    m = Minuit(lsq, mu=0, sigma=1, A=max(c))
    m.migrad()
    m.hesse()
    print(m)
    mu    = m.values["mu"]
    std = m.values["sigma"]
    #Plot the gaussian fitted
    x=np.linspace(bins[0], bins[-1], 300)
    p=norm.pdf(x, mu, std)
    ax[toysGenFrom].plot(x, p*len(r)*np.diff(bins)[0], 'r--', linewidth=2)
    ax[toysGenFrom].text(x=0.95, y=0.95, s=f"N={len(r)}", ha="right", va="top", transform=ax[toysGenFrom].transAxes)
#    ax[toysGenFrom].text(x=0.95, y=0.8, s=f"Fit=MultiPdf\nToy={pdfLabels[cat][toysGenFrom]}", ha="right", va="top", transform=ax[toysGenFrom].transAxes)
    ax[toysGenFrom].text(x=0.05, y=0.95, s=f"$\mu$={mu:.3f}\n$\sigma$={std:.2f}", ha="left", va="top", transform=ax[toysGenFrom].transAxes, color='red')

    if abs(mu) > 0.5:
        ax[toysGenFrom].set_facecolor((1,0,0,0.1))
    elif abs(mu) > 0.2:
        ax[toysGenFrom].set_facecolor((1,1,0,0.1))

    ax[toysGenFrom].set_xlabel(label)

pictureName = f"/t3home/gcelotto/ggHbb/WSFit/output/cat{cat}/plots/multipdf_results/multipdf_envelope_toysGen_matrix_cat{cat}_Signal{expectedSignal}.png"
fig.savefig(pictureName, bbox_inches='tight')
print("Saved in ", pictureName)
# %%

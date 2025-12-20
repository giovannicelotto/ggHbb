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
parser.add_argument("--signalExpected", type=int, help="Signal expected", default=0)
parser.add_argument("--npdf", type=int, help="Signal expected", default=3)
args = parser.parse_args()

cat = args.cat
expectedSignal = args.signalExpected
pdfLabels = {0:["Bern2", "Expo1", "PolExpo3"],
             10:["Bern2", "Bern3", "Expo1", "PolExpo2"],
             100:["Bern2","Expo1","PolExpo2"]}
pdfFitValues = list(np.arange(args.npdf))
toysGenValues = list(np.arange(args.npdf))
fig, ax = plt.subplots(len(pdfFitValues), len(toysGenValues), figsize=(20,20))

def gauss(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

ext=f"Signal{expectedSignal}"
for toysGenFrom in toysGenValues:
    for pdfFit in pdfFitValues:
        fileName = f"/t3home/gcelotto/ggHbb/WSFit/datacards/fitDiagnosticsfitPdf{pdfFit}_ToysGen{toysGenFrom}_{ext}_cat{cat}.root"
        if not os.path.exists(fileName):
            print(f"[WARNING] File {fileName} does not exist, skipping")
            continue
            
        file = uproot.open(fileName)

        
        tree = file["tree_fit_sb"]
        branches = tree.arrays(library="np")
        r = branches["r"]
        rHiErr = branches["rHiErr"]
        rLoErr = branches["rLoErr"]
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
        bins = np.linspace(-7, 7, 30)

        c=ax[toysGenFrom, pdfFit].hist(observable, bins=bins, histtype="stepfilled", alpha=0.5)[0]
        centers = 0.5 * (bins[1:] + bins[:-1])
        errors = np.sqrt(c)  
        lsq = LeastSquares(centers, c, errors, gauss)

        # --- Fit ---
        m = Minuit(lsq, mu=0, sigma=1, A=max(c))
        #m.errordef = Minuit.LEAST_SQUARES
        m.migrad()
        m.hesse()
        mu    = m.values["mu"]
        std = m.values["sigma"]


        # Fit gaussian with iminuit


        mu, std = norm.fit(observable[(observable>bins[0]) & (observable<bins[-1])])
        #Plot the gaussian fitted
        x=np.linspace(bins[0], bins[-1], 300)
        p=norm.pdf(x, mu, std)
        ax[toysGenFrom, pdfFit].plot(x, p*len(r)*np.diff(bins)[0], 'r--', linewidth=2)
        ax[toysGenFrom, pdfFit].text(x=0.95, y=0.95, s=f"N={len(r)}", ha="right", va="top", transform=ax[toysGenFrom, pdfFit].transAxes)
        ax[toysGenFrom, pdfFit].text(x=0.95, y=0.8, s=f"Fit={pdfLabels[cat][pdfFit]}\nToy={pdfLabels[cat][toysGenFrom]}", ha="right", va="top", transform=ax[toysGenFrom, pdfFit].transAxes)
        #ax[toysGenFrom, pdfFit].text(x=0.05, y=0.95, s=f"Mean={observable.mean():.3f}\nStd={observable.std():.3f}", ha="left", va="top", transform=ax[toysGenFrom, pdfFit].transAxes)
        ax[toysGenFrom, pdfFit].text(x=0.05, y=0.95, s=f"$\mu$={mu:.3f}\n$\sigma$={std:.2f}", ha="left", va="top", transform=ax[toysGenFrom, pdfFit].transAxes, color='red')

        if abs(mu) > 0.5:
            ax[toysGenFrom, pdfFit].set_facecolor((1,0,0,0.1))
        elif abs(mu) > 0.2:
            ax[toysGenFrom, pdfFit].set_facecolor((1,1,0,0.1))

        ax[toysGenFrom, pdfFit].set_xlabel(label)

pictureName = f"/t3home/gcelotto/ggHbb/WSFit/output/cat{cat}/plots/multipdf_results/multipdf_toysGenFit_matrix_cat{cat}_{ext}.png"
fig.savefig(pictureName, bbox_inches='tight')
print("Saved in ", pictureName)
# %%

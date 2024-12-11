import numpy as np
from array import array
import os
import sys
#from correctionlib import _core
import correctionlib
import matplotlib.pyplot as plt

filepath = os.path.abspath(__file__)
datapath = os.path.dirname(os.path.dirname(filepath))

import optparse
parser = optparse.OptionParser()
parser.add_option("-y", dest = "year", help ="era")
parser.add_option("--wp", dest = "wp", help = "WP")
parser.add_option("--tagger", dest = "tagger", help = "tagger")
parser.add_option("-v", dest = "v", help = "version", default = "1")
(opts, args) = parser.parse_args()

sfDir = os.path.join(".", "data", "UL"+opts.year)

#fixedWPSFjson = _core.CorrectionSet.from_file(os.path.join(sfDir, "btagging.json"))
fixedWPSFjson = correctionlib.CorrectionSet.from_file(os.path.join(sfDir, "ctagging.json"))
settings = {
    "TnP": [5],
    "wcharm": [4],
    "incl": [0]
    }
labels = {
    5: {
        "corr":  "TnP",
        "color": "blue",
        "label": "b-jet SF"
        },
    4: {
        "corr":  "wcharm",
        "color": "red",
        "label": "c-jet SF"
        },
    0: {
        "corr": "incl",
        "color": "blue",
        "label": "light-jet SF"
        },
    }

outdir = os.path.join("plots", "UL"+opts.year)
if not os.path.exists(outdir):
    os.mkdir(outdir)
ptMax = 300.
pTvalues = np.arange(20., ptMax, 1.)
where = [False for _ in pTvalues]
for setting in settings:
    fixedWPSF = fixedWPSFjson[f"{opts.tagger}_wp"]
    plt.plot([0., ptMax], [1., 1.], color = "black", linestyle = "-", markersize = 0)
    yMax = 1
    yMin = 1
    for flav in settings[setting]:
        corr = labels[flav]["corr"]
        systematics = [""]
        values = {}
        
        sf = fixedWPSF.evaluate("central", corr, opts.wp, flav, 0.5, pTvalues)
        sf_up = {}
        sf_dn = {}
        for s in systematics:
            sf_up[s] = fixedWPSF.evaluate("up"+s, corr, opts.wp, flav, 0.5, pTvalues)
            sf_dn[s] = fixedWPSF.evaluate("down"+s, corr, opts.wp, flav, 0.5, pTvalues)


        # central value
        plt.plot(pTvalues, sf, 
            color = labels[flav]["color"], 
            label = labels[flav]["label"],
            linestyle  = "-", 
            linewidth  = 2, 
            markersize = 0)
        yMax = max(max(sf), yMax)
        yMin = min(min(sf), yMin)
        for s in systematics:
            plt.fill_between(pTvalues, sf_dn[s], sf_up[s], 
                alpha = 0.3,
                label = labels[flav]["label"]+" unc "+s.replace("_",""))
            yMax = max(yMax, max(sf_up[s]), max(sf_dn[s]))
            yMin = min(yMin, min(sf_up[s]), min(sf_dn[s]))
    plt.legend()
    plt.grid(True)
    plt.ylim((yMin-0.1,yMax+0.1))
    plt.xlim((0.,ptMax))

    plt.title(f"AK4 c-tagging, WP: {opts.wp} ({opts.year})", loc = "right")
    plt.title("CMS Simulation Preliminary", loc = "left")
    plt.ylabel("scale factor")
    plt.xlabel("jet pT / GeV")

    plt.savefig(os.path.join(outdir, f"{opts.tagger}_fixedWPSFc_{setting}_{opts.year}_{opts.wp}_v{opts.v}.pdf"))
    plt.savefig(os.path.join(outdir, f"{opts.tagger}_fixedWPSFc_{setting}_{opts.year}_{opts.wp}_v{opts.v}.png"))
    plt.clf()



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
parser.add_option("-y", dest = "year", help ="era", action = "append")
parser.add_option("--wp", dest = "wp", help = "WP")
parser.add_option("--tagger", dest = "tagger", help = "tagger")
parser.add_option("-v", dest = "v", help = "version", default = "1")
(opts, args) = parser.parse_args()

sfDirs = {}
fixedWPSFjson = {}
for y in opts.year:
    sfDirs[y] = os.path.join(".", "data", "UL"+y)

    #fixedWPSFjson = _core.CorrectionSet.from_file(os.path.join(sfDir, "btagging.json"))
    fixedWPSFjson[y] = correctionlib.CorrectionSet.from_file(os.path.join(sfDirs[y], "btagging.json.gz"))

settings = {
    "incl": {
        "default": None,
        },
    "mujets": {
        "default": None,
        },
    "comb": {
        "default": None,
        }
    }
labels = {
    "incl": {
        "flav": 0,
        "color": "blue",
        "label": "inclSF light"
        },
    "mujets": {
        "flav": 5,
        "color": "red",
        "label": "mujetsSF b/c"
        },
    "comb": {
        "flav": 5,
        "color": "red",
        "label": "combSF b/c"
        }
    }
styles = {
    "2016preVFP": "-",
    "2016postVFP": "-.",
    }
outdir = os.path.join("plots", "comparison")
if not os.path.exists(outdir):
    os.mkdir(outdir)
ptMax = 500.
pTvalues = np.arange(20., ptMax, 1.)
where = [False for _ in pTvalues]
for setting in settings:
    for syst in settings[setting]:
        
        if settings[setting][syst] is None: 
            systematics = [""]
        else:
            systematics = [f"_{source}" for source in settings[setting][syst]]
        values = {}
        
        flav = labels[setting]["flav"]

        sf = {}
        sf_up = {}
        sf_dn = {}
        for y in opts.year:
            fixedWPSF = fixedWPSFjson[y][f"{opts.tagger}_{setting}"]
            sf[y] = fixedWPSF.evaluate("central", opts.wp, flav, 0.5, pTvalues)
            sf_up[y] = {}
            sf_dn[y] = {}
            for s in systematics:
                sf_up[y][s] = fixedWPSF.evaluate("up"+s, opts.wp, flav, 0.5, pTvalues)
                sf_dn[y][s] = fixedWPSF.evaluate("down"+s, opts.wp, flav, 0.5, pTvalues)


        plt.title(f"AK4 b-tagging, WP: {opts.wp}", loc = "right")
        plt.title("CMS Simulation Preliminary", loc = "left")
        plt.plot([0., ptMax], [1., 1.], color = "black", linestyle = "-", markersize = 0)
        maxVal = []
        minVal = []
        for y in opts.year:
            # central value
            plt.plot(pTvalues, sf[y], 
                color = labels[setting]["color"], 
                label = labels[setting]["label"]+" ("+y+")",
                linestyle  = styles[y], 
                linewidth  = 2, 
                markersize = 0)
            yMax = max(sf[y])
            yMin = min(sf[y])
            for s in systematics:
                plt.fill_between(pTvalues, sf_dn[y][s], sf_up[y][s], 
                    alpha = 0.3,
                    label = "unc "+s.replace("_","")+" ("+y+")")
                yMax = max(yMax, max(sf_up[y][s]), max(sf_dn[y][s]))
                yMin = min(yMin, min(sf_up[y][s]), min(sf_dn[y][s]))
            maxVal.append(yMax)
            minVal.append(yMin)

        yMin = min(minVal)
        yMax = max(maxVal)
        plt.legend()
        plt.grid(True)
        plt.ylim((yMin-0.1,yMax+0.1))
        plt.xlim((0.,ptMax))
        plt.ylabel("scale factor")
        plt.xlabel("jet pT / GeV")

        plt.savefig(os.path.join(outdir, f"{opts.tagger}_fixedWPSF_{setting}_{opts.wp}_{syst}_v{opts.v}.pdf"))
        plt.savefig(os.path.join(outdir, f"{opts.tagger}_fixedWPSF_{setting}_{opts.wp}_{syst}_v{opts.v}.png"))
        plt.clf()



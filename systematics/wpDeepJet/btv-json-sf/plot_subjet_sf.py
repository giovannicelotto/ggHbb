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

SFjson = correctionlib.CorrectionSet.from_file(os.path.join(sfDir, "subjet_tagging.json"))
settings = {
    "incl": {
        "default": None,
        },
    "lt": {
        "default": None,
        },
    }
labels = {
    "incl": {
        "flav": 0,
        "color": "blue",
        "label": "inclSF light"
        },
    "lt": {
        "flav": 5,
        "color": "red",
        "label": "lifetimeSF b/c"
        },
    }

outdir = os.path.join("plots", "UL"+opts.year)
if not os.path.exists(outdir):
    os.mkdir(outdir)
ptMax = 500.
pTvalues = np.arange(20., ptMax, 1.)
for setting in settings:
    cset = SFjson[f"{opts.tagger}_subjet"]
    for syst in settings[setting]:
        
        if settings[setting][syst] is None: 
            systematics = [""]
        else:
            systematics = [f"_{source}" for source in settings[setting][syst]]
        values = {}
        
        flav = labels[setting]["flav"]
        sf = cset.evaluate("central", setting, opts.wp, flav, 0.5, pTvalues)
        sf_up = {}
        sf_dn = {}
        for s in systematics:
            sf_up[s] = cset.evaluate("up"+s, setting, opts.wp, flav, 0.5, pTvalues)
            sf_dn[s] = cset.evaluate("down"+s, setting, opts.wp, flav, 0.5, pTvalues)


        plt.title(f"AK8 subjet tagging, WP: {opts.wp} ({opts.year})", loc = "right")
        plt.title("CMS Simulation Preliminary", loc = "left")
        plt.plot([0., ptMax], [1., 1.], color = "black", linestyle = "-", markersize = 0)
        # central value
        plt.plot(pTvalues, sf, 
            color = labels[setting]["color"], 
            label = labels[setting]["label"],
            linestyle  = "-", 
            linewidth  = 2, 
            markersize = 0)
        yMax = max(1., max(sf))
        yMin = min(1., min(sf))
        for s in systematics:
            plt.fill_between(pTvalues, sf_dn[s], sf_up[s], 
                alpha = 0.3,
                label = "unc "+s.replace("_",""))
            yMax = max(yMax, max(sf_up[s]), max(sf_dn[s]))
            yMin = min(yMin, min(sf_up[s]), min(sf_dn[s]))
        plt.legend()
        plt.grid(True)
        plt.ylim((yMin-0.1,yMax+0.1))
        plt.xlim((0.,ptMax))
        plt.ylabel("scale factor")
        plt.xlabel("jet pT / GeV")

        plt.savefig(os.path.join(outdir, f"{opts.tagger}_subjetSF_{setting}_{opts.year}_{opts.wp}_{syst}_v{opts.v}.pdf"))
        plt.savefig(os.path.join(outdir, f"{opts.tagger}_subjetSF_{setting}_{opts.year}_{opts.wp}_{syst}_v{opts.v}.png"))
        plt.clf()



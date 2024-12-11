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
parser.add_option("--tagger", dest = "tagger", help = "tagger")
parser.add_option("-v", dest = "v", help = "version", default = "1")
(opts, args) = parser.parse_args()

sfDir = os.path.join(".", "data", "UL"+opts.year)

SFjson = correctionlib.CorrectionSet.from_file(os.path.join(sfDir, "ctagging.json"))
SF = SFjson[f"{opts.tagger}_shape"]

settings = {
    "light": {
        "flav": 0,
        "label": "light-jets",
        "cvl": [0.05, 0.1, 0.15, 0.25, 0.4, 0.7, 0.9],
        },
    "b": {
        "flav": 5,
        "label": "b-jets",
        "cvl": [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9],
        },
    "c": {
        "flav": 4,
        "label": "c-jets",
        "cvl": [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9],
        },
    }

outdir = os.path.join("plots", "UL"+opts.year)
if not os.path.exists(outdir):
    os.mkdir(outdir)
x = np.arange(0.0, 1.0, 0.01)
for setting in settings:
    flav = settings[setting]["flav"]
    label = settings[setting]["label"]

    plt.title(f"AK4 c-tagging itFit ({opts.year}) {label}", loc = "right")
    plt.title("CMS Simulation Preliminary", loc = "left")

    plt.plot([-0.0, 1.0], [1., 1.], color = "black", linestyle = "-", markersize = 0)
        
    yMin = 1.
    yMax = 1.
    for cvl in settings[setting]["cvl"]:
        sfs = SF.evaluate("central", flav, cvl, x)
        plt.plot(x, sfs, 
            label = "CvL={:.2f}".format(cvl),
            linestyle = "-",
            linewidth = 2,
            markersize = 0)
        yMin = min(min(sfs), yMin)
        yMax = max(max(sfs), yMax)

    plt.legend()
    plt.grid(True)
    plt.ylim((yMin-0.1,yMax+0.1))
    plt.xlim((-0.0,1.1))
    plt.ylabel("scale factor")
    plt.xlabel(f"{opts.tagger} CvB value")

    plt.savefig(os.path.join(outdir, f"{opts.tagger}_ctagging_itFit_{setting}_{opts.year}_v{opts.v}.pdf"))
    plt.savefig(os.path.join(outdir, f"{opts.tagger}_ctagging_itFit_{setting}_{opts.year}_v{opts.v}.png"))
    plt.clf()



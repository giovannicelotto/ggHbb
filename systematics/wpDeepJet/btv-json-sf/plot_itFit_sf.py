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

SFjson = correctionlib.CorrectionSet.from_file(os.path.join(sfDir, "btagging.json"))
SF = SFjson[f"{opts.tagger}_shape"]

settings = {
    "light": {
        "pt":  [25., 35., 50., 80., 200.],
        "eta": [0., 1., 2.],
        "flav": 0,
        "label": "light-jets",
        "x": [0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6],
        },
    "b": {
        "pt":  [25., 40., 60., 80., 120., 200.],
        "eta": [0.],
        "flav": 5,
        "label": "b-jets",
        "x": [-0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
    "c": {
        "pt":  [100.],
        "eta": [0.],
        "flav": 4,
        "label": "c-jets",
        "x": [0.5]
        },
    }

outdir = os.path.join("plots", "UL"+opts.year)
if not os.path.exists(outdir):
    os.mkdir(outdir)
x = np.arange(0., 1., 0.01)
ptval = np.arange(20., 200., 1.)
for setting in settings:
    flav = settings[setting]["flav"]
    label = settings[setting]["label"]
    for eta in settings[setting]["eta"]:
        plt.title(f"AK4 b-tagging itFit ({opts.year}) {label}", loc = "right")
        plt.title("CMS Simulation Preliminary", loc = "left")

        plt.plot([-0.1, 1.1], [1., 1.], color = "black", linestyle = "-", markersize = 0)
        
        yMin = 1.
        yMax = 1.
        for pt in settings[setting]["pt"]:
            sfs = SF.evaluate("central", flav, eta, pt, x)
            plt.plot(x, sfs, 
                label = "eta={:.1f}, pt={:.0f}".format(eta, pt),
                linestyle = "-",
                linewidth = 2,
                markersize = 0)
            yMin = min(min(sfs), yMin)
            yMax = max(max(sfs), yMax)

        plt.legend()
        plt.grid(True)
        plt.ylim((yMin-0.1,yMax+0.1))
        plt.xlim((-0.1,1.1))
        plt.ylabel("scale factor")
        plt.xlabel(f"{opts.tagger} b-tag value")

        plt.savefig(os.path.join(outdir, f"{opts.tagger}_itFit_{setting}_{opts.year}_eta{eta:.1f}_disc_v{opts.v}.pdf"))
        plt.savefig(os.path.join(outdir, f"{opts.tagger}_itFit_{setting}_{opts.year}_eta{eta:.1f}_disc_v{opts.v}.png"))
        plt.clf()

        '''
        plt.title(f"AK4 b-tagging itFit ({opts.year}) {label}", loc = "right")
        plt.title("CMS Simulation Preliminary", loc = "left")

        plt.plot([0., 200.], [1., 1.], color = "black", linestyle = "-", markersize = 0)
        
        yMin = 1.
        yMax = 1.
        for disc in settings[setting]["x"]:
            sfs = SF.evaluate("central", flav, eta, ptval, disc)
            plt.plot(ptval, sfs, 
                label = "eta={:.1f}, discr={:.2f}".format(eta, disc),
                linestyle = "-",
                linewidth = 2,
                markersize = 0)
            yMin = min(min(sfs), yMin)
            yMax = max(max(sfs), yMax)

        plt.legend()
        plt.grid(True)
        plt.ylim((yMin-0.1,yMax+0.1))
        plt.xlim((0.,200.))
        plt.ylabel("scale factor")
        plt.xlabel(f"{opts.tagger} p_T / GeV")

        plt.savefig(os.path.join(outdir, f"{opts.tagger}_itFit_{setting}_{opts.year}_eta{eta:.1f}_pt_v{opts.v}.pdf"))
        plt.savefig(os.path.join(outdir, f"{opts.tagger}_itFit_{setting}_{opts.year}_eta{eta:.1f}_pt_v{opts.v}.png"))
        plt.clf()      
        '''



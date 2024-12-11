############################################
# based on ipynb provided by Xavier Coubez #
############################################

import numpy as np
import os
import sys

import optparse
parser = optparse.OptionParser()
parser.add_option("--year","-y",dest="year",
    choices = ["2018", "2017", "2016preVFP", "2016postVFP"],
    help = "year to evaluate")
parser.add_option("--wp","-w",dest="wp",
    choices = ["L", "M", "T"],
    help = "working point to evaluate")
parser.add_option("--flavor","-f",dest="flavor",
    choices = [0,4,5],
    help = "flavor to evaluate")
parser.add_option("--method","-m",dest="method",
    choices = ["incl", "mujets", "comb"],
    help = "sf method to evaluate")
(opts, args) = parser.parse_args()

# test the output
from correctionlib import _core

# test read out
evaluator_wp = _core.CorrectionSet.from_file(f'./UL{opts.year}_wp_temp.json')


# signature for working point is evaluate('systematic', 'method', 'working_point', 'flavor', 'abseta', 'pt')
valsf_deepCSV = evaluator_wp["deepCSV_wp"].evaluate("central", "comb", 0, 0, 0., 30.)
print(f"deepCSV sf is: {valsf_deepCSV:.3f}")

valsf_deepJet = evaluator_wp["deepJet_wp"].evaluate("central", "comb", 0, 0, 0., 30.)
print(f"deepJet sf is: {valsf_deepJet:.3f}")



# test read out
evaluator_corr = _core.CorrectionSet.from_file(f'./UL{opts.year}_corr_temp.json')


# signature for working point is evaluate('systematic', 'method', 'working_point', 'flavor', 'abseta', 'pt')
valsf_deepCSV = evaluator_corr["deepCSV_wp"].evaluate("central", "comb", 0, 0, 0., 30.)
print(f"deepCSV sf is: {valsf_deepCSV:.3f}")

valsf_deepJet = evaluator_corr["deepJet_wp"].evaluate("central", "comb", 0, 0, 0., 30.)
print(f"deepJet sf is: {valsf_deepJet:.3f}")


evaluator_bd = _core.CorrectionSet.from_file(f'./UL{opts.year}_breakdown_temp.json')
# signature for working point is evaluate('systematic', 'method', 'working_point', 'flavor', 'abseta', 'pt')
valsf_deepCSV = evaluator_bd["deepCSV_wp"].evaluate("central", "comb", 0, 0, 0., 30.)
print(f"deepCSV sf is: {valsf_deepCSV:.3f}")

valsf_deepJet = evaluator_bd["deepJet_wp"].evaluate("central", "comb", 0, 0, 0., 30.)
print(f"deepJet sf is: {valsf_deepJet:.3f}")



import matplotlib.pyplot as plt
pts = np.linspace(20., 500., 1000)
dJ_default  = [evaluator_wp["deepJet_wp"].evaluate("central", "mujets", 0, 0, 0., x) for x in pts]
dJ_yearcorr = [evaluator_corr["deepJet_wp"].evaluate("central", "mujets", 0, 0, 0., x) for x in pts]
#dJ_bd = [evaluator_bd["deepJet_wp"].evaluate("central", "comb", 0, 0, 0., x) for x in pts]
plt.plot(pts, dJ_default, label = "default deepJet")
plt.plot(pts, dJ_yearcorr, label = "yearCorr deepJet")
#plt.plot(pts, dJ_bd, label = "catBreakdown deepJet")
plt.legend()
plt.xlabel("pt")
plt.ylabel("SFb")
plt.title(f"deepJet UL{opts.year} - mujets")
plt.savefig(f"deepJet_central_mujets_b_UL{opts.year}.pdf")
plt.clf()

dC_default  = [evaluator_wp["deepCSV_wp"].evaluate("central", "mujets", 0, 0, 0., x) for x in pts]
dC_yearcorr = [evaluator_corr["deepCSV_wp"].evaluate("central", "mujets", 0, 0, 0., x) for x in pts]
print(dC_default[0:30])
print(dC_yearcorr[0:30])
plt.plot(pts, dC_default, label = "default deepCSV")
plt.plot(pts, dC_yearcorr, label = "yearCorr deepCSV")
plt.legend()
plt.xlabel("pt")
plt.ylabel("SFb")
plt.title(f"deepCSV UL{opts.year} - mujets")
plt.savefig(f"deepCSV_central_mujets_b_UL{opts.year}.pdf")
plt.clf()

dJ_default  = [evaluator_wp["deepJet_wp"].evaluate("central", "comb", 0, 0, 0., x) for x in pts]
dJ_yearcorr = [evaluator_corr["deepJet_wp"].evaluate("central", "comb", 0, 0, 0., x) for x in pts]
#dJ_bd = [evaluator_bd["deepJet_wp"].evaluate("central", "comb", 0, 0, 0., x) for x in pts]
plt.plot(pts, dJ_default, label = "default deepJet")
plt.plot(pts, dJ_yearcorr, label = "yearCorr deepJet")
#plt.plot(pts, dJ_bd, label = "catBreakdown deepJet")
plt.legend()
plt.xlabel("pt")
plt.ylabel("SFb")
plt.title(f"deepJet UL{opts.year} - comb")
plt.savefig(f"deepJet_central_comb_b_UL{opts.year}.pdf")
plt.clf()

dC_default  = [evaluator_wp["deepCSV_wp"].evaluate("central", "comb", 0, 0, 0., x) for x in pts]
dC_yearcorr = [evaluator_corr["deepCSV_wp"].evaluate("central", "comb", 0, 0, 0., x) for x in pts]
print(dC_default[0:30])
print(dC_yearcorr[0:30])
plt.plot(pts, dC_default, label = "default deepCSV")
plt.plot(pts, dC_yearcorr, label = "yearCorr deepCSV")
plt.legend()
plt.xlabel("pt")
plt.ylabel("SFb")
plt.title(f"deepCSV UL{opts.year} - comb")
plt.savefig(f"deepCSV_central_comb_b_UL{opts.year}.pdf")
plt.clf()

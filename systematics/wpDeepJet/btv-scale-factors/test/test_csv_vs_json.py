import correctionlib
import pandas as pd
import numpy as np

import optparse
parser = optparse.OptionParser()
parser.add_option("--wp", dest="wp", default="M")
parser.add_option("--flav", dest="flav", type=int, default=5)
parser.add_option("--syst", dest="syst", default="central")
parser.add_option("--method", dest="method", default="comb")
parser.add_option("--tagger", dest="tagger", default="particleNet")
(opts, args) = parser.parse_args()

cset = correctionlib.CorrectionSet.from_file(
    "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2022_Summer22/btagging.json.gz")

csv_file = f"../2022_Summer22/csv/btagging_fixedWP/{opts.tagger}_v0.csv"
df = pd.read_csv(csv_file)

# test comb SFs
c = cset[f"{opts.tagger}_{opts.method}"]
df = df[df["type"] == opts.method]

def eval_csv(syst, wp, flav, eta, x):
    # extract the line from CSV file
    sdf = df[ (df["syst"] == syst) & \
            (df["wp"] == wp) & \
            (df["flav"] == flav) & \
            (df["etaMin"] <= eta) & \
            (df["etaMax"] > eta) & \
            (df["ptMin"] <= x) & \
            (df["ptMax"] > x) ]

    if len(sdf) != 1: 
        print(f"Couldnt find unique SF in csv file")
        raise ValueError(sdf)

    # evauate formula with x = pt
    formula = sdf["formula"].iloc[0].replace("log", "np.log")
    return eval(formula)

print(f"SFs for {opts.tagger}: {opts.syst} at {opts.wp} with flav={opts.flav}")
for pt in np.linspace(50, 950, 10):
    # get sf from json
    jsf = c.evaluate(opts.syst, opts.wp, opts.flav, 0., pt)
    # get sf from csv
    csf = eval_csv(opts.syst, opts.wp, opts.flav, 0., pt)
    print(f"pt={pt:.0f}, jsonSF={jsf:.4f}, csvSF={csf:.4f}, diff={jsf-csf:.4f}")

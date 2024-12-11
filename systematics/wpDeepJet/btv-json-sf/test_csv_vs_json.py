import correctionlib
import pandas as pd
import numpy as np
import common

import optparse
parser = optparse.OptionParser()
parser.add_option("--wp", dest="wp", default="M")
parser.add_option("--flav", dest="flav", type=int, default=5)
parser.add_option("--syst", dest="syst", default="central")
parser.add_option("--method", dest="method", default="comb")
parser.add_option("--tagger", dest="tagger", default="particleNet")
parser.add_option("-x",dest="x",default=0.5,type=float,help="discr value")
parser.add_option("--json", dest="json")
parser.add_option("--csv", dest="csv")
(opts, args) = parser.parse_args()

cset = correctionlib.CorrectionSet.from_file(
    opts.json)

csv_file = opts.csv
df = pd.read_csv(csv_file, skipinitialspace=True)
df = common.rename_columns(df, opts.tagger)
print(df)

# test comb SFs
c = cset[f"{opts.tagger}_{opts.method}"]

new = True
if not "type" in df.columns:
    new = False
if opts.method == "shape":
    opts.wp = "shape"

if new:
    df = df[df["type"] == opts.method.replace("shape","iterativefit")]
else:
    df = df[df["measurementType"] == opts.method.replace("shape","iterativefit")]
print(df)

def eval_csv(syst, wp, flav, eta, pt, x):
    # extract the line from CSV file
    print(syst, wp, flav, eta, pt, x)
    if new:
        sdf = df[ (df["syst"] == syst) & \
                (df["wp"] == wp) & \
                (df["flav"] == flav) & \
                (df["etaMin"] <= eta) & \
                (df["etaMax"] > eta) & \
                (df["discrMin"] <= x) & \
                (df["discrMax"] > x) & \
                (df["ptMin"] <= pt) & \
                (df["ptMax"] > pt) ]
    else:
        sdf = df[ (df["sysType"] == syst) & \
                (df["OperatingPoint"] == wp) & \
                (df["jetFlavor"] == flav) & \
                (df["etaMin"] <= eta) & \
                (df["etaMax"] > eta) & \
                (df["discrMin"] <= x) & \
                (df["discrMax"] > x) & \
                (df["ptMin"] <= pt) & \
                (df["ptMax"] > pt) ]


    if len(sdf) != 1: 
        print(f"Couldnt find unique SF in csv file")
        raise ValueError(sdf)

    # evauate formula with x = pt
    formula = sdf["formula"].iloc[0].replace("log", "np.log")
    return eval(formula)

print(f"SFs for {opts.tagger}: {opts.syst} at {opts.wp} with flav={opts.flav}")
for pt in np.linspace(50, 950, 10):
    # get sf from json
    if opts.method == "shape":
        jsf = c.evaluate(opts.syst, int(opts.flav), 0., pt, opts.x)
    else:
        jsf = c.evaluate(opts.syst, opts.wp, int(opts.flav), 0., pt)
    # get sf from csv
    csf = eval_csv(opts.syst, opts.wp, opts.flav, 0., pt, opts.x)
    print(f"pt={pt:.0f}, jsonSF={jsf:.4f}, csvSF={csf:.4f}, diff={jsf-csf:.4f}")

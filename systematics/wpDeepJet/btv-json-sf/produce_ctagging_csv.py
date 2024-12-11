import pandas as pd
import ROOT

import optparse
parser = optparse.OptionParser()
parser.add_option("-r", dest="inputRootFile")

(opts, args) = parser.parse_args()

f = ROOT.TFile.Open(opts.inputRootFile)
keys = [k.GetName() for k in f.GetListOfKeys()]

def filterHists(hists):
    new_hists = []
    for h in hists:
        if "TotalUnc" in h:
            continue
        if "ValuesSystOnly" in h:
            continue
        if "withMaxUncs" in h:
            continue
        new_hists.append(h)
    return new_hists

lHists = filterHists([k for k in keys if k.startswith("SFl")])
bHists = filterHists([k for k in keys if k.startswith("SFb")])
cHists = filterHists([k for k in keys if k.startswith("SFc")])
print(lHists)
print(bHists)
print(cHists)

out = {
    "OperatingPoint": [],
    "measurementType": [],
    "sysType": [],
    "jetFlavor": [],
    "cvbMin": [],
    "cvbMax": [],
    "cvlMin": [],
    "cvlMax": [],
    "formula": []
    }
    

def get_hist_sf(f, key, flav, name, out):
    h = f.Get(key)
    for ix in range(h.GetNbinsX()):
        xMin = h.GetXaxis().GetBinLowEdge(ix+1)
        xMax = h.GetXaxis().GetBinLowEdge(ix+2)
        for iy in range(h.GetNbinsY()):
            v = h.GetBinContent(ix+1, iy+1)
            yMin = h.GetYaxis().GetBinLowEdge(iy+1)
            yMax = h.GetYaxis().GetBinLowEdge(iy+2)
            out["OperatingPoint"].append("shape")
            out["measurementType"].append("iterativefit")
            out["sysType"].append(name)
            out["jetFlavor"].append(flav)
            out["cvlMin"].append(round(xMin,3))
            out["cvlMax"].append(round(xMax,3))
            out["cvbMin"].append(round(yMin,3))
            out["cvbMax"].append(round(yMax,3))
            out["formula"].append(round(v,10))

def get_flav_hists(f, hists, flav, base, out):
    for k in hists:
        if k == base:
            name = "central"
        else:
            name = k.replace(base+"_","")
            if name.endswith("Up"):
                name = "up_"+name.replace("Up","")
            if name.endswith("Down"):
                name = "down_"+name.replace("Down","")
        get_hist_sf(f, k, flav, name, out)
    

get_flav_hists(f, lHists, 0, "SFl_hist", out)
get_flav_hists(f, cHists, 4, "SFc_hist", out)
get_flav_hists(f, bHists, 5, "SFb_hist", out)

df = pd.DataFrame.from_dict(out)
out_file = opts.inputRootFile.replace(".root",".csv")
df.to_csv(out_file, index = False, 
    columns = ["OperatingPoint", "measurementType", "sysType", "jetFlavor", "cvlMin", "cvlMax", "cvbMin", "cvbMax", "formula"]
    )
print(f"created output file {out_file}")

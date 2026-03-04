# %%
import ROOT
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)
ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT.gErrorIgnoreLevel = ROOT.kWarning
import yaml
import sys
import argparse
import mplhep as hep
# Style
hep.style.use("CMS")
ROOT.gROOT.SetBatch(True)


# %%
# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', type=str, default="0", help='Config File')
args = parser.parse_args([]) if hasattr(sys, 'ps1') or not sys.argv[1:] else parser.parse_args()

# %%
import uproot
location_of_fit = "/t3home/gcelotto/ggHbb/tt_CR/workspace_NNqm"
f = uproot.open("%s/fitDiagnosticscat%s.root"%(location_of_fit, args.category))
SF_NN = f["tree_fit_sb"].arrays()["SF_NN"][0]
SF_NNLoErr = f["tree_fit_sb"].arrays()["SF_NNLoErr"][0]
SF_NNHiErr = f["tree_fit_sb"].arrays()["SF_NNHiErr"][0]


# %%

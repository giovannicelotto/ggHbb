# Read here
# https://muon-wiki.docs.cern.ch/guidelines/corrections/#medium-pt-id-efficiencies

path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/nonResonant/ttbar_noTight//ttbar2L2Nu"
import glob
import numpy as np
import pandas as pd
fileNames = glob.glob(path+"/*.parquet")
df = pd.read_parquet(fileNames)
den = (df.muon_pt>9) & (abs(df.muon_eta)<1.5) & (df.dijet_pt>100) & (df.Muon_tt_pt>0)
num = (den) & (df.Muon_tt_dxy < 0.2) & (df.Muon_tt_dz < 0.5)
print(num.sum()/den.sum())


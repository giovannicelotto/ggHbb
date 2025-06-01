# run_getStdAllBins.py

import argparse
import pickle
import numpy as np
import os
import sys

# Add module paths if needed
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new")
from helpersABCD.getStdBinABCD_bootstrap import getStdAllBins

parser = argparse.ArgumentParser()
parser.add_argument('--var', type=str, required=True)
parser.add_argument('--bin_idx', type=int, required=True)
args = parser.parse_args()

# Load arguments
base_path = '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/doubleDisco/Apr01_1000p0/temp_data'
with open(f"{base_path}/dfData_{args.var}.pkl", 'rb') as f:
    dfData = pickle.load(f)
with open(f"{base_path}/dfMC_{args.var}.pkl", 'rb') as f:
    dfMC = pickle.load(f)
with open(f"{base_path}/args_to_save_{args.var}.pkl", 'rb') as f:
    saved_args = pickle.load(f)

# Extract relevant arguments
xx = saved_args['xx']
m_fit = saved_args['m_fit']
q_fit = saved_args['q_fit']
t1 = saved_args['t1']
t2 = saved_args['t2']
var = saved_args['var']
bins = saved_args['bins']
idx = args.bin_idx

low = bins[idx]
high = bins[idx + 1]

output_path = f"/t3home/gcelotto/ggHbb/abcd/new/output/std_bin_jobs/std_SR_{var}_bin{idx}.npy"

# Call the function
getStdAllBins(dfData, dfMC, xx, np.array([low, high]), m_fit, q_fit, t1, t2, save=True, path=output_path)

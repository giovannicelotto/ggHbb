# %%
from functions import loadMultiParquet_Data_new, loadMultiParquet_v2, getCommonFilters, getDfProcesses_v2, cut
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %%
dfs, sumw = loadMultiParquet_v2(paths=[35], nMCs=-1, returnNumEventsTotal=True, filters=getCommonFilters(btagTight=True))

# %%
dfs_ = cut(dfs, 'jet1_btagTight', 0.5, None)
dfs_ = cut(dfs_, 'jet2_btagTight', 0.5, None)
#dfs_ = cut(dfs_, 'dijet_pt', 160, None)
df = pd.concat(dfs_)
# %%
df
# %%
df['weight']=df.genWeight * df.btag_central * df.sf * df.PU_SF * 5300 /sumw[0]
# %%
df.weight.sum()*1000*41.6
# %%

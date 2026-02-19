# %%
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functions import loadMultiParquet_v2, getCommonFilters
# %%
df = loadMultiParquet_v2(paths=[37], nMCs=-1, columns=None, returnNumEventsTotal=False, returnFileNumberList=False, filters=getCommonFilters(btagWP="L", cutDijet=True, ttbarCR=False))[0]
# %%
inclusive_den = df.flat_weight.sum()
inclusive_num = df[df.jet2_has_trigMuon==1].flat_weight.sum()
TT_num = df[(df.jet2_has_trigMuon==1) & (df.jet1_btagDeepFlavB>0.71) & (df.jet2_btagDeepFlavB>0.71)].flat_weight.sum()
TT_den = df[(df.jet1_btagDeepFlavB>0.71) & (df.jet2_btagDeepFlavB>0.71) ].flat_weight.sum()
MM_num = df[(df.jet2_has_trigMuon==1) & (df.jet1_btagDeepFlavB>0.2783) & (df.jet2_btagDeepFlavB>0.2783) & ~((df.jet1_btagDeepFlavB>0.71) & (df.jet2_btagDeepFlavB>0.71))].flat_weight.sum()
MM_den = df[(df.jet1_btagDeepFlavB>0.2783) & (df.jet2_btagDeepFlavB>0.2783) & ~((df.jet1_btagDeepFlavB>0.71) & (df.jet2_btagDeepFlavB>0.71))].flat_weight.sum()
LL_num = df[(df.jet2_has_trigMuon==1) & ~((df.jet1_btagDeepFlavB>0.2783) & (df.jet2_btagDeepFlavB>0.2783))].flat_weight.sum()
LL_den = df[~((df.jet1_btagDeepFlavB>0.2783) & (df.jet2_btagDeepFlavB>0.2783))].flat_weight.sum()

eff_inclusive = inclusive_num/inclusive_den
eff_TT = TT_num/TT_den
eff_MM = MM_num/MM_den
eff_LL = LL_num/LL_den
# %%
print(f"Inclusively : {eff_inclusive*100:.2f}% +- {np.sqrt(eff_inclusive*(1-eff_inclusive)/inclusive_den)*100:.3f} of events have a trigMuon in jet2")
print(f"TT region : {eff_TT*100:.2f}% +- {np.sqrt(eff_TT*(1-eff_TT)/TT_den)*100:.3f} of events have a trigMuon in jet2")
print(f"MM region : {eff_MM*100:.2f}% +- {np.sqrt(eff_MM*(1-eff_MM)/MM_den)*100:.3f} of events have a trigMuon in jet2")
print(f"LL region : {eff_LL*100:.2f}% +- {np.sqrt(eff_LL*(1-eff_LL)/LL_den)*100:.3f} of events have a trigMuon in jet2")

# %%
from plotFeatures import plotNormalizedFeatures
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import loadMultiParquet_v2, getCommonFilters, getDfProcesses_v2, loadMultiParquet_Data
# %%
dfs = loadMultiParquet_v2(paths=[0,
                                 1,
                                 2,3,4,
                                 5,6,7,8,9,10,11,12,13,14,15,16,17,18,
                                 19,20,21,22,35], nMCs=10,columns=None, returnNumEventsTotal=False, selectFileNumberList=None, returnFileNumberList=False, filters=getCommonFilters())

dfsData = loadMultiParquet_Data(paths=[0], nReals=1)

# %%
for idx in range(len(dfs)):
    processName = getDfProcesses_v2()[0].process[idx]
    plotNormalizedFeatures([dfsData[0][0], dfs[idx]], outFile="/t3home/gcelotto/ggHbb/outputs/plots/features/features_%s.png"%processName, legendLabels=['Data', processName], weights=None, colors=['blue', 'red'], figsize=(30, 45))
# %%

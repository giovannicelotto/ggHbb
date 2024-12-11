# %%
from functions import loadMultiParquet_v2, loadMultiParquet_Data
# %%
nReal = 1
nMC = [10, 10]
predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_nov18"
columns = ['dijet_mass',       
          #'jet1_btagDeepFlavB',   'jet2_btagDeepFlavB',
          #'PU_SF', 'sf', 
          ]
# %%
dfs, numEventsList, fileNumberList = loadMultiParquet_v2(paths=[0, 1], nMCs=nMC, columns=columns, returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)
# %%
dfs, lumi_tot, fileNumberList = loadMultiParquet_Data(paths=[0, 1, 2], nReals=[-1, -1, -1], columns=columns, selectFileNumberList=None, returnFileNumberList=True)
# %%

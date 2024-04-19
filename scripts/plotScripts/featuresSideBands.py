import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotFeatures import plotNormalizedFeatures
from utilsForPlot import loadParquet

realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others"
signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others"

signal, realData = loadParquet(signalPath, realDataPath, nSignalFiles=-1, nRealDataFiles=20, columns=None, fileNumbers=None, returnNumEventsTotal=False)

mSignal_peak = (signal.dijet_mass > 125-2*17) & (signal.dijet_mass < 125+2*17)
mData_side = (realData.dijet_mass < 125-2*17) | (realData.dijet_mass > 125+2*17)
plotNormalizedFeatures(data=[signal[mSignal_peak], realData[mData_side], realData[~ (mData_side)]],
                           outFile = "/t3home/gcelotto/ggHbb/outputs/plots/features/Features_sidebands.png",
                           legendLabels = ['Signal SR', 'BParking Side', 'BParking SR'] ,
                           colors = ['blue', 'red', 'green'],
                           figsize=(15, 30))
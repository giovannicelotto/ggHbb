import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
#from getFeaturesBScoreBased import getFeaturesBScoreBased
from utilsForPlot import getBins, scatter2Features, loadData, getFeaturesBScoreBased

# Loading files
signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures"
realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/withMoreFeatures"

signal, realData = loadData(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=100, nRealDataFiles=9)
labels = getFeaturesBScoreBased()
bins=getBins()
scatter2Features(signal, realData, labels=labels, bins=bins, outFile="/t3home/gcelotto/ggHbb/outputs/plots/scatterPlot2DNew.png")




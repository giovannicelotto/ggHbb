import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utilsForPlot import  loadParquet
from functions import getXSectionBR
import sys, glob
import mplhep as hep
hep.style.use("CMS")
flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
signalPath = flatPathCommon +"/GluGluHToBB/**"
realDataPath = flatPathCommon +"/Data1A/**"
nReal = 1008
signal, realData, numEventsTotal= loadParquet(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=nReal, columns=['dijet_mass', 'sf'], returnNumEventsTotal=True)
fig, ax = plt.subplots(1, 1)
hep.cms.label(lumi=round(float(0.774*nReal/1017), 4), ax=ax)
bins = np.linspace(0, 2000, 300)

ax.hist(realData.dijet_mass, bins=bins, histtype=u'step', weights=realData.sf)
ax.set_yscale('log')
outName="/t3home/gcelotto/ggHbb/outputs/plots/massSpectrum/YmesonMass.png"
fig.savefig(outName, bbox_inches='tight')
print(outName)
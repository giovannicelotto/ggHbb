import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import sys
from matplotlib.ticker import AutoMinorLocator, LogLocator

totalNanoEntries = 190153970

signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/old"
realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData"

signalFileNames = glob.glob(signalPath+"/*.npy")[:]
bscore5FileNames = glob.glob(realDataPath+"/*bScoreBased5_*.npy")[:]
bscore2FileNames = glob.glob(realDataPath+"/*bScoreBased2_*.npy")[:]
bscore3FileNames = glob.glob(realDataPath+"/*bScoreBased3_*.npy")[:]
bscore4FileNames = glob.glob(realDataPath+"/*bScoreBased4_*.npy")[:]




print("%d files for MC ggHbb" %len(signalFileNames))
print("%d files for BPH 2018" %len(bscore5FileNames))
print("%d files for leading Dijet" %len(bscore2FileNames))
print("%d files for bscore3FileNames" %len(bscore3FileNames))
print("%d files for bscore4FileNames" %len(bscore4FileNames))

signal = np.load(signalFileNames[0])[:,19]
for signalFileName in signalFileNames[1:]:
    sys.stdout.write('\r')
    sys.stdout.write("   %d/%d   "%(signalFileNames.index(signalFileName)+1, len(signalFileNames)))
    sys.stdout.flush()
    currentSignal = np.load(signalFileName)[:,19]
    signal = np.concatenate((signal, currentSignal))
print("Signal shape: ", signal.shape)

bscore2 = np.load(bscore2FileNames[0])[:,17]
for bscore2FileName in bscore2FileNames[1:]:
    sys.stdout.write('\r')
    sys.stdout.write("   %d/%d   "%(bscore2FileNames.index(bscore2FileName)+1, len(bscore2FileNames)))
    sys.stdout.flush()
    currentbscore2 = np.load(bscore2FileName)[:,17]
    bscore2 = np.concatenate((bscore2, currentbscore2))
print("bscore2 shape: ", bscore2.shape)

bscore3 = np.load(bscore3FileNames[0])[:,17]
for bscore3FileName in bscore3FileNames[1:]:
    sys.stdout.write('\r')
    sys.stdout.write("   %d/%d   "%(bscore3FileNames.index(bscore3FileName)+1, len(bscore3FileNames)))
    sys.stdout.flush()
    currentBscore3 = np.load(bscore3FileName)[:,17]
    bscore3 = np.concatenate((bscore3, currentBscore3))
print("bscore3 shape: ", bscore3.shape)

bscore4 = np.load(bscore4FileNames[0])[:,17]
for bscore4FileName in bscore4FileNames[1:]:
    sys.stdout.write('\r')
    sys.stdout.write("   %d/%d   "%(bscore4FileNames.index(bscore4FileName)+1, len(bscore4FileNames)))
    sys.stdout.flush()
    currentBscore4 = np.load(bscore4FileName)[:,17]
    bscore4 = np.concatenate((bscore4, currentBscore4))
print("bscore4 shape: ", bscore4.shape)
    
bscore5 = np.load(bscore5FileNames[0])[:,17]
for bscore5FileName in bscore5FileNames[1:]:
    sys.stdout.write('\r')
    sys.stdout.write("   %d/%d   "%(bscore5FileNames.index(bscore5FileName)+1, len(bscore5FileNames)))
    sys.stdout.flush()
    #print(bscore5FileName)
    currentbscore5 = np.load(bscore5FileName)[:,17]
    bscore5 = np.concatenate((bscore5, currentbscore5))
print("Real data shape: ", bscore5.shape)


fig, ax = plt.subplots(1, 1)
bins = np.linspace(1, 500, 500)
bscore5Counts = np.histogram(bscore5[:], bins=bins)[0]
signalCounts = np.histogram (signal[:], bins=bins)[0]
bscore2Counts = np.histogram (bscore2[:], bins=bins)[0]
bscore3Counts = np.histogram(bscore3[:], bins=bins)[0]
bscore4Counts = np.histogram(bscore4[:], bins=bins)[0]
realLumi = 0.67*np.sum(bscore2Counts)/totalNanoEntries


N_mini = np.load("/t3home/gcelotto/bbar_analysis/output/N_mini.npy")


bscore5Counts = bscore5Counts*totalNanoEntries/np.sum(bscore5Counts)
bscore2Counts = bscore2Counts*totalNanoEntries/np.sum(bscore2Counts)
bscore3Counts = bscore3Counts*totalNanoEntries/np.sum(bscore3Counts)
bscore4Counts = bscore4Counts*totalNanoEntries/np.sum(bscore4Counts)
signalCounts = signalCounts/N_mini*30.52*0.67*1000

ax.hist(bins[:-1], bins=bins, weights=signalCounts, color='blue', histtype=u'step', label='MC ggHbb')
ax.hist(bins[:-1], bins=bins, weights=bscore2Counts, color='green', histtype=u'step', label='BParking Data n=2')
ax.hist(bins[:-1], bins=bins, weights=bscore3Counts, color='pink', histtype=u'step', label='BParking Data n=3')
ax.hist(bins[:-1], bins=bins, weights=bscore4Counts, color='orange', histtype=u'step', label='BParking Data n=4')
ax.hist(bins[:-1], bins=bins, weights=bscore5Counts, color='red', histtype=u'step', label='BParking Data n=5')

ax.set_xlabel("Dijet Mass [GeV]", fontsize=16)
ax.set_ylabel("Events / %.1f GeV"%(bins[1]-bins[0]), fontsize=16)
#ax.text(x=0.5, y = 0.5, s="Selected dijets:", fontsize=14, transform=fig.transFigure)
#ax.text(x=0.5, y = 0.45, s="Max DeepFlavB sum", fontsize=14, transform=fig.transFigure)
ax.text(s=r"%.5f fb$^{-1}$ (13 TeV)"%realLumi, x=1.00, y=1.02, ha='right', fontsize=12,  transform=ax.transAxes, **{'fontname':'Arial'})
ax.set_yscale('log')
#ax.grid()
ax.tick_params(which='major', length=8)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(LogLocator(subs='all'))
ax.tick_params(which='minor', length=4)
ax.legend(loc='best')
fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/dijetMass.pdf", bbox_inches='tight')

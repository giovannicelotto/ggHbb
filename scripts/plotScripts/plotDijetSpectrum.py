import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import sys
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import norm, crystalball
from scipy.integrate import quad
from utilsForPlot import loadData, loadDataOnlyMass, getXSectionBR
import mplhep as hep
hep.style.use("CMS")

afterCut, log = True, True
if afterCut:
    try:
        signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/afterCutWithMoreFeatures/signalCut.npy"
        realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/afterCutWithMoreFeatures/realDataCut.npy"
        signal = np.load(signalPath)[:,21]
        realData = np.load(realDataPath)[:,21]
    except:
        print("Taken")
        signalPath ="/t3home/gcelotto/ggHbb/outputs/signalCut.npy"
        realDataPath="/t3home/gcelotto/ggHbb/outputs/realDataCut.npy"
        signal = np.load(signalPath)[:,21]
        realData = np.load(realDataPath)[:,21]
else:

    # Loading files
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/withMoreFeatures"

    signal, realData = loadDataOnlyMass(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=40)

# Correction factors and counters
N_SignalMini = np.load("/t3home/gcelotto/bbar_analysis/output/N_mini.npy")
N_DataNano = np.load("/t3home/gcelotto/ggHbb/outputs/N_BPH_Nano.npy")   
correctionSignal = 1/N_SignalMini*getXSectionBR()*0.67*1000
correctionData = N_DataNano/len(realData)*0.883
#correctionData = 1017/300
if afterCut:
    print("Watchout")
    correctionData = N_DataNano/len(realData)*0.883*0.0008896
    correctionData = 1017/407
    #
# Counts numbers of data in 1A for signal and bkg
totalSignalCounts = 0
totalData = 0                   


# Plot
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(6, 7))
fig.subplots_adjust(hspace=0.0)
fig.align_ylabels([ax,ax2])

x1, x2, nbin = 0, 300, 101
ax.set_xlim(x1, x2)
bins = np.linspace(x1, x2, nbin)
if log:
    ax.set_xlim(x1, x2)
    ax.set_yscale('log')

# Prepare the points
signalCounts = np.histogram (np.clip(signal, bins[0], bins[-1]), bins=bins)[0]
realDataCounts = np.histogram(np.clip(realData, bins[0], bins[-1]), bins=bins)[0]
# Get the error    
realDataCountsErr= np.sqrt(realDataCounts)*correctionData
signalCountsErr= np.sqrt(signalCounts)*correctionSignal #if log==True else np.sqrt(signalCounts)*correctionSignal*10**4
#Normalize data
signalCounts = signalCounts*correctionSignal #if log==True else signalCounts*correctionSignal*10**4
realDataCounts = realDataCounts*correctionData
#Keep track of all the expected entries at 1A lumi
totalData += np.sum(realDataCounts)
totalSignalCounts+=np.sum(signalCounts)
# Plot data
ax.hist(bins[:-1], bins=bins, weights=signalCounts, color='blue', histtype=u'step', label='MC ggHbb')#$\times10^4$
ax.hist(bins[:-1], bins=bins, weights=realDataCounts, color='red', histtype=u'step', label='BParking Data')

# Fit
x_data_tot = (bins[:-1]+bins[1:])/2
if afterCut is False:
    x1_fr, x2_fr = 50, 165
    gauss=False
else:
    x1_fr, x2_fr = 95, 170
    gauss=True
print("Fit region limited from %d to %d"%(x1_fr, x2_fr))
fit_region = (x_data_tot>x1_fr) & (x_data_tot<x2_fr)
y_fr = signalCounts[fit_region]
x_fr = x_data_tot[fit_region]

if gauss:
    def gaussian_function(x, amplitude, mean, sigma):
        return amplitude * norm.pdf(x, loc=mean, scale=sigma)   
    initial_guess = [5*10**2, 125, 20]  # Initial guess for parameters
    params, covariance = curve_fit(gaussian_function, x_fr, y_fr, p0=initial_guess)
    amplitude_fit, mean_fit, sigma_fit,  = params
    y_fit = gaussian_function(x_data_tot, amplitude_fit, mean_fit, sigma_fit)
    result, abserr = quad(gaussian_function, x1_fr, x2_fr, args=(amplitude_fit, mean_fit, sigma_fit))[:2]

else:
    def crystal_ball_function(x, amplitude, mean, sigma, alpha, n):
        return amplitude * crystalball.pdf(x, alpha, n, loc=mean, scale=sigma)
    initial_guess = [3.56*10**3, 125, 20, 1, 10]  # Initial guess for parameters
    params, covariance = curve_fit(crystal_ball_function, x_fr, y_fr, sigma = signalCountsErr[fit_region],p0=initial_guess)
    amplitude_fit, mean_fit, sigma_fit, alpha_fit, n_fit = params
    y_fit = crystal_ball_function(x_data_tot, amplitude_fit, mean_fit, sigma_fit, alpha_fit, n_fit)

    num_samples=10
    amplitude_samples = np.random.normal(amplitude_fit, np.sqrt(covariance[0, 0]), num_samples)
    mean_samples = np.random.normal(mean_fit, np.sqrt(covariance[1, 1]), num_samples)
    sigma_samples = np.random.normal(sigma_fit, np.sqrt(covariance[2, 2]), num_samples)
    alpha_samples = np.random.normal(alpha_fit, np.sqrt(covariance[3, 3]), num_samples)
    n_samples = np.random.normal(n_fit, np.sqrt(covariance[4, 4]), num_samples)
    for i in range(num_samples):
        print(amplitude_samples[i], mean_samples[i], sigma_samples[i], alpha_samples[i], n_fit)
        result, _ = quad(crystal_ball_function, x1_fr, x2_fr, args=(amplitude_samples[i], mean_samples[i], sigma_samples[i], alpha_samples[i], n_fit))[:2]
        print(result)

    result, abserr= quad(crystal_ball_function, x1_fr, x2_fr, args=(amplitude_fit, mean_fit, sigma_fit, alpha_fit, n_fit))[:2]
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
ax.plot(x_data_tot, y_fit, label='Fitted Curve', color='red')

# Second plot
ax2.hlines(y=0, xmin=x1, xmax=x2, color='black')
residuals = (signalCounts-y_fit)/signalCountsErr
ax2.errorbar(x_data_tot, residuals, yerr=1, color='black', linestyle=' ', marker='o')
ax2.set_ylim(-5, 5)

for par, var in zip(params, np.diag(covariance)):
    print(par, " +- ", np.sqrt(var))
ax.text(s=r"N$=%d\pm%d$"%(result/(bins[1]-bins[0]), abserr/(bins[1]-bins[0])), x=0.07, y=0.8,  ha='left', transform=ax.transAxes, fontsize=12)
ax.text(s=r"$\mu=%.2f\pm%.2f$"%(mean_fit, np.sqrt(covariance[1, 1])),          x=0.07, y=0.75,  ha='left', transform=ax.transAxes, fontsize=12)
ax.text(s=r"$\sigma=%.2f\pm%.2f$"%(sigma_fit, np.sqrt(covariance[2, 2])),      x=0.07, y=0.7,  ha='left', transform=ax.transAxes, fontsize=12)
if gauss is False:
    ax.text(s=r"$\alpha=%.2f\pm%.2f$"%(alpha_fit, np.sqrt(covariance[3, 3])),      x=0.07, y=0.65,  ha='left', transform=ax.transAxes, fontsize=12)
    ax.text(s=r"n=$%.1f\pm%.1f$"%(n_fit, np.sqrt(covariance[4, 4])),      x=0.07, y=0.6,  ha='left', transform=ax.transAxes, fontsize=12)
chi2 = np.sum(((y_fit[fit_region]-y_fr)/signalCountsErr[fit_region])**2)

ndof = (len(y_fr)-len(params))
ax.text(s=r"$\chi^2$/n$_\mathrm{dof}=%d/%d$"%(chi2, ndof),                      x=0.07, y=0.45,  ha='left', transform=ax.transAxes, fontsize=12)

# Temporary check
#fig2, ax2=plt.subplots(1, 1)
#ax2.plot(x_data, y_data, label='Data', marker='o', color='black', linestyle=None)
#ax2.plot(x_data, crystal_ball_function(x_data, 5*10**7, 125, 20, 1, 2))
#ax2.legend()
#fig2.savefig("/t3home/gcelotto/ggHbb/outputs/plots/temp.png", bbox_inches='tight')
#End



# EXPECTED SIGNIFICANCE
    #How many events between 95 and 165 GeV
x1_sb, x2_sb  = mean_fit - 2*sigma_fit, mean_fit + 2*sigma_fit
print(x1_sb, x2_sb)
maskSignal = (signal[:]>x1_sb) & (signal[:]<x2_sb)
maskData = (realData[:]>x1_sb) & (realData[:]<x2_sb)
    
S = np.sum(maskSignal)*correctionSignal
B = np.sum(maskData)*correctionData
SErr = np.sqrt(np.sum(maskSignal))*correctionSignal
BErr = np.sqrt(np.sum(maskData))*correctionData
    
Scommon_exponent = int("{:e}".format(SErr).split('e')[1])
Bcommon_exponent = int("{:e}".format(BErr).split('e')[1])
    
Scommon_exponent = 0 if Scommon_exponent<0 else Scommon_exponent
ax.text(s="0.67 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02,  ha='right', transform=ax.transAxes, fontsize=16)
if Scommon_exponent==0:
    ax.text(s=r"S(2$\sigma$) = %d $\pm$ %d"%(S/(10**Scommon_exponent), SErr/(10**Scommon_exponent)),                                     x=0.93, y=0.45,  ha='right', transform=ax.transAxes, fontsize=12)
else:
    ax.text(s=r"S(2$\sigma$) = (%d $\pm$ %d) $\times e%d$"%(S/(10**Scommon_exponent), SErr/(10**Scommon_exponent), Scommon_exponent),      x=0.93, y=0.45,  ha='right', transform=ax.transAxes, fontsize=12)
ax.text(s=r"B(2$\sigma$) = (%d $\pm$ %d) $\times 10^{%d}$"%(B/(10**Bcommon_exponent), BErr/(10**Bcommon_exponent), Bcommon_exponent),      x=0.93, y=0.4,  ha='right', transform=ax.transAxes, fontsize=12)
ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.1e"%(S/(np.sqrt(B))),                                                                      x=0.93, y=0.35,  ha='right', transform=ax.transAxes, fontsize=12)
ax.text(s="S/B = %.1e"%(S/B),                                                                                                   x=0.93, y=0.3,  ha='right', transform=ax.transAxes, fontsize=12)
# @ Full lumi
ax.text(s="@Full BPH Lumi (41.6 fb$^{-1}$)",                                                                                      x=0.93, y=0.15,  ha='right', transform=ax.transAxes, fontsize=12)
ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.1e"%(S*np.sqrt(41.6/0.67)/(np.sqrt(B))),                                                     x=0.93, y=0.1,  ha='right', transform=ax.transAxes, fontsize=12)



ax2.set_xlabel("Dijet Mass [GeV]", fontsize=14)




for i in range(len(bins)-1):
    if i ==0 :
        rect = patches.Rectangle((bins[i], realDataCounts[i] - realDataCountsErr[i]),
                bins[i+1]-bins[i], 2 *  realDataCountsErr[i],
                linewidth=0, edgecolor='red', facecolor='none', hatch='///') #label='Uncertainty')
    else:
        rect = patches.Rectangle((bins[i], realDataCounts[i] - realDataCountsErr[i]),
                bins[i+1]-bins[i], 2 *  realDataCountsErr[i],
                linewidth=0, edgecolor='red', facecolor='none', hatch='///')
    ax.add_patch(rect)

for i in range(len(bins)-1):
    if i ==0 :
        rect = patches.Rectangle((bins[i], signalCounts[i]*10**0 - signalCountsErr[i]*10**0),
                bins[i+1]-bins[i], 2 *  signalCountsErr[i]*10**0,
                linewidth=0, edgecolor='blue', facecolor='none', hatch='///') #label='Uncertainty')
    else:
        rect = patches.Rectangle((bins[i], signalCounts[i]*10**0 - signalCountsErr[i]*10**0),
                bins[i+1]-bins[i], 2 *  signalCountsErr[i]*10**0,
                linewidth=0, edgecolor='blue', facecolor='none', hatch='///')
    ax.add_patch(rect)
if not log:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax.vlines(x=[x1_fr, x2_fr], ymin=0, ymax=ax.get_ylim()[1], color='blue', alpha=0.2, label='Fit region')
ax.vlines(x=[x1_sb, x2_sb], ymin=0, ymax=ax.get_ylim()[1], color='green', alpha=0.2, label=r'$\mu\pm2\sigma$')
ax.legend(loc='upper right', fontsize=16)

ax.set_ylabel("Events / %.1f GeV"%(bins[1]-bins[0]), fontsize=16)
ax2.set_ylabel(r"$\frac{\mathrm{Signal Events - Signal Fit}}{\mathrm{Signal Events Err}}$", fontsize=16)


print(totalSignalCounts)
print(totalData)
if log:
    outName = "/t3home/gcelotto/ggHbb/outputs/plots/dijetMassCut_log.png" if afterCut else "/t3home/gcelotto/ggHbb/outputs/plots/dijetMass_log.png"
else:
    outName = "/t3home/gcelotto/ggHbb/outputs/plots/dijetMassCut.png" if afterCut else "/t3home/gcelotto/ggHbb/outputs/plots/dijetMass.png"
print("Saving in ", outName)
fig.savefig(outName, bbox_inches='tight')


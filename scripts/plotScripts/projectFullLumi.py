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
import sys
import mplhep as hep
hep.style.use("CMS")

def plotDijetMass(afterCut= True, log = True, fit = True):
    if afterCut:
        print("Opening files after the cuts...")
        signalPath = "/t3home/gcelotto/ggHbb/outputs/signalCut.npy"
        realDataPath = "/t3home/gcelotto/ggHbb/outputs/realDataCut.npy"
        signal = np.load(signalPath)[:,21]
        realData = np.load(realDataPath)[:,21]

    else:
        # Loading files
        signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withMoreFeatures"
        realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData/withMoreFeatures"

        signal, realData = loadDataOnlyMass(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=-1)


    currentLumi = 0.774*20/1017
    N_SignalMini = np.load("/t3home/gcelotto/ggHbb/outputs/counters/N_mini.npy")
    #N_DataNano = np.load("/t3home/gcelotto/ggHbb/outputs/counters/N_BPH_Nano.npy")   
    correctionSignal = 1/N_SignalMini*getXSectionBR()*41.6/currentLumi*1000
    correctionData = 1017/20 # projection to 0.774
    correctionData = correctionData*41.6/0.774

    totalSignalCounts = 0
    totalData = 0                   


    # Plot
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(6, 7))
    fig.subplots_adjust(hspace=0.0)
    fig.align_ylabels([ax,ax2])

    x1, x2, nbin = 0, 300, 101
    ax.set_xlim(x1, x2)
    bins = np.linspace(x1, x2, nbin)
    ax.set_xlim(x1, x2)
    if log:
        ax.set_yscale('log')

    # Prepare the points
    signalCounts = np.histogram (np.clip(signal, bins[0], bins[-1]), bins=bins)[0]
    realDataCounts = np.histogram(np.clip(realData, bins[0], bins[-1]), bins=bins)[0]
    # Get the error    
    realDataCountsErr= np.sqrt(realDataCounts)*correctionData
    signalCountsErr= np.sqrt(signalCounts)*correctionSignal if log==True else np.sqrt(signalCounts)*correctionSignal
    #Normalize data
    signalCounts = signalCounts*correctionSignal if log==True else signalCounts*correctionSignal
    realDataCounts = realDataCounts*correctionData
    #Keep track of all the expected entries at 1A lumi
    totalData += np.sum(realDataCounts)
    totalSignalCounts+=np.sum(signalCounts)
    # Plot data
    ax.hist(bins[:-1], bins=bins, weights=signalCounts, color='blue', histtype=u'step', label='MC ggHbb')
    ax.hist(bins[:-1], bins=bins, weights=realDataCounts, color='red', histtype=u'step', label='BParking Data')
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
    # Fit
    
    if (fit):
        x_data_tot = (bins[:-1]+bins[1:])/2
        x1_br, x2_br = 90, 260
        x1_sr, x2_sr = 110, 140
        
        # fit of the background
        def exponential_decay(x, A, tau, C):
            return A * np.exp(-x / tau) + C
        bkg_fit_region = ((x_data_tot>x1_br)&(x_data_tot<x1_sr))|((x_data_tot<x2_br)&(x_data_tot>x2_sr))
        y_br = realDataCounts[bkg_fit_region]
        x_br = x_data_tot[bkg_fit_region]
        initial_guess = [5*10**7, 47, 100]  # Initial guess for parameters
        print("Fit region limited from %d to %d"%(x1_br, x2_br))
        params, covariance = curve_fit(exponential_decay, x_br, y_br, p0=initial_guess)
        amplitude_fit_br, lambda_fit_br, const_fit_br,  = params
        y_fit_br = exponential_decay(x_data_tot, amplitude_fit_br, lambda_fit_br, const_fit_br)
        
        
        #fit_region = (x_data_tot>x1_sr) & (x_data_tot<x2_sr)
        #y_sr = signalCounts[fit_region]
        #x_sr = x_data_tot[fit_region]
#
        #
        #def exp_plus_gaussian_function(x, amplitude, mean, sigma):
        #    return amplitude * norm.pdf(x, loc=mean, scale=sigma)   
        #initial_guess = [5*10**2, 125, 20]  # Initial guess for parameters
        #params, covariance = curve_fit(gaussian_function, x_fr, y_fr, p0=initial_guess)
        #amplitude_fit, mean_fit, sigma_fit,  = params
        #y_fit = gaussian_function(x_data_tot, amplitude_fit, mean_fit, sigma_fit)
        #result, abserr = quad(gaussian_function, x1_fr, x2_fr, args=(amplitude_fit, mean_fit, sigma_fit))[:2]

        

        ax.plot(x_data_tot, y_fit_br+signalCounts, label='Fitted Curve + Signal', color='Blue')
        ax.plot(x_data_tot, y_fit_br, label='Fitted Curve', color='red')

        # Second plot

        residuals = (y_br-y_fit_br[bkg_fit_region])/realDataCountsErr[bkg_fit_region]
        ax2.errorbar(x_data_tot[bkg_fit_region], residuals, yerr=1, color='black', linestyle=' ', marker='o')


        for par, var in zip(params, np.diag(covariance)):
            print(par, " +- ", np.sqrt(var))
        #ax.text(s=r"N$=%d\pm%d$"%(result/(bins[1]-bins[0]), abserr/(bins[1]-bins[0])), x=0.07, y=0.8,  ha='left', transform=ax.transAxes, fontsize=12)
        #ax.text(s=r"$\mu=%.2f\pm%.2f$"%(mean_fit, np.sqrt(covariance[1, 1])),          x=0.07, y=0.75,  ha='left', transform=ax.transAxes, fontsize=12)
        #ax.text(s=r"$\sigma=%.2f\pm%.2f$"%(sigma_fit, np.sqrt(covariance[2, 2])),      x=0.07, y=0.7,  ha='left', transform=ax.transAxes, fontsize=12)
        #if gauss is False:
        #    ax.text(s=r"$\alpha=%.2f\pm%.2f$"%(alpha_fit, np.sqrt(covariance[3, 3])),      x=0.07, y=0.65,  ha='left', transform=ax.transAxes, fontsize=12)
        #    ax.text(s=r"$\mathrm{n}=%.1f\pm%.1f$"%(n_fit, np.sqrt(covariance[4, 4])),      x=0.07, y=0.6,  ha='left', transform=ax.transAxes, fontsize=12)
        chi2 = np.sum(((y_fit_br[bkg_fit_region]-y_br)/realDataCountsErr[bkg_fit_region])**2)
        print(chi2, "/", np.sum(bkg_fit_region))
        #ndof = (len(y_fr)-len(params))
        #ax.text(s=r"$\chi^2$/n$_\mathrm{dof}=%d/%d$"%(chi2, ndof),                      x=0.07, y=0.45,  ha='left', transform=ax.transAxes, fontsize=12)

        # Temporary check
        #fig2, ax2=plt.subplots(1, 1)
        #ax2.plot(x_data, y_data, label='Data', marker='o', color='black', linestyle=None)
        #ax2.plot(x_data, crystal_ball_function(x_data, 5*10**7, 125, 20, 1, 2))
        #ax2.legend()
        #fig2.savefig(outFolder+"/temp.png", bbox_inches='tight')
        #End



        # EXPECTED SIGNIFICANCE
            #How many events between 95 and 165 GeV
        #x1_sb, x2_sb  = mean_fit - 2*sigma_fit, mean_fit + 2*sigma_fit
        #print(x1_sb, x2_sb)
        #maskSignal = (signal[:]>x1_sb) & (signal[:]<x2_sb)
        #maskData = (realData[:]>x1_sb) & (realData[:]<x2_sb)
#
        #S = np.sum(maskSignal)*correctionSignal
        #B = np.sum(maskData)*correctionData
        #SErr = np.sqrt(np.sum(maskSignal))*correctionSignal
        #BErr = np.sqrt(np.sum(maskData))*correctionData
#
        #Scommon_exponent = int("{:e}".format(SErr).split('e')[1])
        #Bcommon_exponent = int("{:e}".format(BErr).split('e')[1])
#
        #Scommon_exponent = 0 if Scommon_exponent<0 else Scommon_exponent
        #if Scommon_exponent==0:
        #    ax.text(s=r"S(2$\sigma$) = %d $\pm$ %d"%(S/(10**Scommon_exponent), SErr/(10**Scommon_exponent)),                                     x=0.93, y=0.45,  ha='right', transform=ax.transAxes, fontsize=12)
        #else:
        #    ax.text(s=r"S(2$\sigma$) = (%d $\pm$ %d) $\times e%d$"%(S/(10**Scommon_exponent), SErr/(10**Scommon_exponent), Scommon_exponent),      x=0.93, y=0.45,  ha='right', transform=ax.transAxes, fontsize=12)
        #ax.text(s=r"B(2$\sigma$) = (%d $\pm$ %d) $\times 10^{%d}$"%(B/(10**Bcommon_exponent), BErr/(10**Bcommon_exponent), Bcommon_exponent),      x=0.93, y=0.4,  ha='right', transform=ax.transAxes, fontsize=12)
        #ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.3f"%(S/(np.sqrt(B))),                                                                      x=0.93, y=0.35,  ha='right', transform=ax.transAxes, fontsize=12)
        #ax.text(s="S/B = %.1e"%(S/B),                                                                                                   x=0.93, y=0.3,  ha='right', transform=ax.transAxes, fontsize=12)
        ## @ Full lumi
        #ax.text(s="@Full BP Lumi (41.6 fb$^{-1}$)",                                                                                      x=0.93, y=0.15,  ha='right', transform=ax.transAxes, fontsize=12)
        #ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.2f"%(S*np.sqrt(41.6/currentLumi)/(np.sqrt(B))),                                                     x=0.93, y=0.1,  ha='right', transform=ax.transAxes, fontsize=12)


    # End of Fit
    ax2.hlines(y=0, xmin=x1, xmax=x2, color='black')
    ax2.set_ylim(-5, 5)
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

    if fit:
        #ax.vlines(x=[x1_br, x2_br, x1_sr, x2_sr], ymin=0, ymax=ax.get_ylim()[1], color='blue', alpha=0.8, label='Fit region')
        ax.fill_between(x_data_tot, np.ones(len(x_data_tot))*ax.get_ylim()[1], where=bkg_fit_region, alpha=0.3, label='Bkg Fit Region')

        #ax.vlines(x=[x1_sb, x2_sb], ymin=0, ymax=ax.get_ylim()[1], color='green', alpha=0.2, label=r'$\mu\pm2\sigma$')
    
    ax.legend(fontsize=16, bbox_to_anchor=(1, 1))

    ax.set_ylabel("Events / %.1f GeV"%(bins[1]-bins[0]), fontsize=16)
    ax2.set_ylabel(r"$\frac{\mathrm{Signal Events - Signal Fit}}{\mathrm{Signal Events Err}}$", fontsize=16)


    print(totalSignalCounts)
    print(totalData)
    outFolder = "/t3home/gcelotto/ggHbb/outputs/plots"
    outName = outFolder+"/projectionFullLumi.png"
    print("Saving in ", outName)
    print(log, outName)
    ax.text(s="%.2f fb$^{-1}$ (13 TeV)"%currentLumi, x=1.00, y=1.02,  ha='right', transform=ax.transAxes, fontsize=16)
    fig.savefig(outName, bbox_inches='tight')

if __name__=="__main__":

    afterCut = int(sys.argv[1]) if len(sys.argv)>1 else True
    log      = int(sys.argv[2]) if len(sys.argv)>2 else False
    fit      = int(sys.argv[3]) if len(sys.argv)>3 else True
    plotDijetMass(afterCut, log, fit)
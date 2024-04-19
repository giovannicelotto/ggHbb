import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import norm, crystalball
from scipy.integrate import quad
from utilsForPlot import  loadParquet
from functions import getXSectionBR
import sys, glob
import mplhep as hep
hep.style.use("CMS")

def cut(data, feature, min, max):
    newData = []
    for df in data:
        if min is not None:
            df = df[df[feature] > min]
        if max is not None:
            df = df[df[feature] < max]
        newData.append(df)
    return newData


def fitFunc(bins, signalCounts, signalCountsErr, gauss, ax):
    x_data_tot = (bins[:-1]+bins[1:])/2
# fit region
    x1_fr, x2_fr = 90, 180
    print("Fit region limited from %d to %d"%(x1_fr, x2_fr))
    fit_region = (x_data_tot>x1_fr) & (x_data_tot<x2_fr)
    y_fr = signalCounts[fit_region]
    x_fr = x_data_tot[fit_region]

    if gauss:
        def gaussian_function(x, amplitude, mean, sigma):
            return amplitude * norm.pdf(x, loc=mean, scale=sigma)
        initial_guess = [5*10**2, 125, 17.1]  # Initial guess for parameters
        params, covariance = curve_fit(gaussian_function, x_fr, y_fr, p0=initial_guess, bounds=([0, 120, 12], [np.inf, 130, 21]))
        amplitude_fit, mean_fit, sigma_fit = params
        y_fit = gaussian_function(x_data_tot, amplitude_fit, mean_fit, sigma_fit)
        result, abserr = quad(gaussian_function, x1_fr, x2_fr, args=(amplitude_fit, mean_fit, sigma_fit))[:2]

    else:
        def crystal_ball_function(x, amplitude, mean, sigma, alpha, n):
            return amplitude * crystalball.pdf(x, alpha, n, loc=mean, scale=sigma)
        initial_guess = [np.max(signalCounts)*100, 125, 17, 0.9, 130]  # Initial guess for parameters
        params, covariance = curve_fit(crystal_ball_function, x_fr, y_fr, sigma = signalCountsErr[fit_region],p0=initial_guess, bounds=([0, 122, 15, 0, 0], [np.inf, 129, 20, 2, 900]) )
        amplitude_fit, mean_fit, sigma_fit, alpha_fit, n_fit = params
        y_fit = crystal_ball_function(x_data_tot, amplitude_fit, mean_fit, sigma_fit, alpha_fit, n_fit)

#                result, abserr= quad(crystal_ball_function, x1_fr, x2_fr, args=(amplitude_fit, mean_fit, sigma_fit, alpha_fit, n_fit))[:2]
    ax.plot(x_data_tot, y_fit, label='Fitted Curve', color='red')
    ax.vlines(x=[x1_fr, x2_fr], ymin=0, ymax=ax.get_ylim()[1], color='blue', alpha=0.2, label='Fit region')
    
    
    for par, var in zip(params, np.diag(covariance)):
        print(par, " +- ", np.sqrt(var))
        #ax.text(s=r"N$=%d\pm%d$"%(result/(bins[1]-bins[0]), abserr/(bins[1]-bins[0])), x=0.04, y=0.95,  ha='left', transform=ax.transAxes, fontsize=12)
    ax.text(s=r"Jet p$_\mathrm{T}>20$ GeV",                                        x=0.05, y=0.75,  ha='left', transform=ax.transAxes, fontsize=24)
    ax.text(s=r"$\mu=%.2f\pm%.2f$"%(mean_fit, np.sqrt(covariance[1, 1])),          x=0.96, y=0.7,  ha='right', transform=ax.transAxes, fontsize=18)
    ax.text(s=r"$\sigma=%.2f\pm%.2f$"%(sigma_fit, np.sqrt(covariance[2, 2])),      x=0.96, y=0.65,  ha='right', transform=ax.transAxes, fontsize=18)
    if gauss is False:
        pass
        #ax.text(s=r"$\alpha=%.2f\pm%.2f$"%(alpha_fit, np.sqrt(covariance[3, 3])),      x=0.96, y=0.65,  ha='right', transform=ax.transAxes, fontsize=18)
        #ax.text(s=r"$\mathrm{n}=%.1f\pm%.1f$"%(n_fit, np.sqrt(covariance[4, 4])),      x=0.96, y=0.6,  ha='right', transform=ax.transAxes, fontsize=18)
    chi2 = np.sum(((y_fit[fit_region]-y_fr)/signalCountsErr[fit_region])**2)

    ndof = (len(y_fr)-len(params))
    ax.text(s=r"$\chi^2$/n$_\mathrm{dof}=%d/%d$"%(chi2, ndof),                      x=0.96, y=0.6,  ha='right', transform=ax.transAxes, fontsize=18)
    return y_fit, params, covariance


def patchError(bins, counts, countsErr, color, ax):
    for i in range(len(bins)-1):
        if i ==0 :
            rect = patches.Rectangle((bins[i], counts[i] - countsErr[i]),
                    bins[i+1]-bins[i], 2 *  countsErr[i],
                    linewidth=0, edgecolor='red', facecolor='none', hatch='///') #label='Uncertainty')
        else:
            rect = patches.Rectangle((bins[i], counts[i] - countsErr[i]),
                    bins[i+1]-bins[i], 2 *  countsErr[i],
                    linewidth=0, edgecolor='red', facecolor='none', hatch='///')
        ax.add_patch(rect)
    return




def plotDijetMass(fit = True, realFiles=1):
    # Loading files
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    signalPath = flatPathCommon +"/GluGluHToBB/**"
    realDataPath = flatPathCommon +"/Data1A/**"
    columnsToRead =['dijet_mass', 'sf', 'jet1_pt', 'jet2_pt']

    signal, realData, numEventsTotal= loadParquet(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=realFiles, columns=columnsToRead, returnNumEventsTotal=True)
    totalDataFlat = len(realData)

# cut    
    signal, realData = cut([signal, realData], 'jet1_pt', 20, None)
    signal, realData = cut([signal, realData], 'jet2_pt', 20, None)
    
# drop unused columns
    columnsToDrop = [x for x in columnsToRead if x not in ['sf', 'dijet_mass']]
    signal, realData = signal.drop(columns=columnsToDrop), realData.drop(columns=columnsToDrop)
    
    lumiPerEvent = np.load("/t3home/gcelotto/ggHbb/outputs/lumiPerEvent.npy")
    x1_sb, x2_sb  = 125.09 - 2*17, 125.09 + 2*17
    maskSignal = (signal.dijet_mass>x1_sb) & (signal.dijet_mass<x2_sb)
    maskData = (realData.dijet_mass>x1_sb) & (realData.dijet_mass<x2_sb)

    S = np.sum(signal.sf[maskSignal])*lumiPerEvent*totalDataFlat/numEventsTotal*getXSectionBR()*1000
    B = np.sum(maskData)
    
    print("Signal 2sigma", S)
    print("Data 2sigma", B)
    print("Sig", S/np.sqrt(B)*np.sqrt(41.6/(lumiPerEvent*totalDataFlat)))        
        
    realFiles = 1017 if realFiles==-1 else realFiles
    currentLumi = lumiPerEvent*totalDataFlat
    
    correctionSignal = 1/numEventsTotal*getXSectionBR()*currentLumi*1000
    correctionData = 1
    visibilityNonLog = 10**4
    visibilityNonLogForZ = 10**3
                  
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    x1, x2, nbin = 0, 300, 101
    ax.set_xlim(x1, x2)
    bins = np.linspace(x1, x2, nbin)

    # Prepare the points
        # Z counts
    signalCounts = np.histogram (np.clip(signal.dijet_mass, bins[0], bins[-1]), weights=signal.sf, bins=bins)[0]
    realDataCounts = np.histogram(np.clip(realData.dijet_mass, bins[0], bins[-1]), bins=bins)[0]

    realDataCountsErr= np.sqrt(realDataCounts)*correctionData
    signalCountsErr= np.sqrt(signalCounts)*correctionSignal*visibilityNonLog
    #Normalize data
    signalCounts = signalCounts*correctionSignal*visibilityNonLog
    realDataCounts = realDataCounts*correctionData


    # Keep track of all the expected entries at 1A lumi
    
    print(np.sum(signalCounts))
    # Plot data
    ax.hist(bins[:-1], bins=bins, weights=realDataCounts, color='red', histtype=u'step', label='BParking Data')
    ax.hist(bins[:-1], bins=bins, weights=signalCounts, color='blue', histtype=u'step', label='Signal')
    ax.set_ylim(ax.get_ylim()[0]+0.01, ax.get_ylim()[1]*1.3)
    
    # Fit
    if (fit):
        gauss=False
        y_fit, params, covariance = fitFunc(bins, signalCounts, signalCountsErr, gauss, ax)
        if gauss:
            amplitude_fit, mean_fit, sigma_fit = params
        else:
            amplitude_fit, mean_fit, sigma_fit, alpha_fit, n_fit = params
        
        print("Total number of signal events this (total) lumi", np.sum(signal.sf)*correctionSignal, np.sum(signal.sf)*correctionSignal*41.6/currentLumi)
        print("Total number of BParking events this (total) lumi", len(realData), len(realData)*41.6/currentLumi)
        x1_sb, x2_sb  = mean_fit - 2*sigma_fit, mean_fit + 2*sigma_fit
        print(x1_sb, x2_sb)
        maskSignal = (signal.dijet_mass[:]>x1_sb) & (signal.dijet_mass[:]<x2_sb)
        maskData = (realData.dijet_mass[:]>x1_sb) & (realData.dijet_mass[:]<x2_sb)

        S = np.sum(signal.sf[maskSignal])*correctionSignal
        B = np.sum(maskData)*correctionData
        SErr = np.sqrt(np.sum(signal.sf[maskSignal]))*correctionSignal
        BErr = np.sqrt(np.sum(maskData))*correctionData


        
        ax.text(s=r"S(2$\sigma$) = %.1f $\pm$ %.1f"%(round(S, 1), round(SErr, 1)),               x=0.96, y=0.45,     ha='right', transform=ax.transAxes, fontsize=18)
        ax.text(s=r"B(2$\sigma$) = %.1f $\pm$ %.1f"%(round(B, 1), round(BErr, 1)),               x=0.96, y=0.4,      ha='right', transform=ax.transAxes, fontsize=18)
        ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.3f"%(S/(np.sqrt(B))),                               x=0.96, y=0.35,     ha='right', transform=ax.transAxes, fontsize=18)
        ax.text(s="S/B = %.1e"%(S/B),                                                            x=0.96, y=0.3,      ha='right', transform=ax.transAxes, fontsize=18)
        
        # @ Full lumi
        ax.text(s="@Full BP Lumi (41.6 fb$^{-1}$)",                                              x=0.96, y=0.17,     ha='right', transform=ax.transAxes, fontsize=18)
        ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.2f"%(S*np.sqrt(41.6/currentLumi)/(np.sqrt(B))),     x=0.96, y=0.12,     ha='right', transform=ax.transAxes, fontsize=18)


    # End of Fit
    ax.set_xlabel("Dijet Mass [GeV]", fontsize=24)
    ax.set_ylabel("Events / %.1f GeV"%(bins[1]-bins[0]), fontsize=24)
    
    patchError(bins, realDataCounts, realDataCountsErr, 'red', ax)
    patchError(bins, signalCounts, signalCountsErr, 'blue', ax)
    
    ax.ticklabel_format(axis='y', style='plain')
    ax.vlines(x=[x1_sb, x2_sb], ymin=0, ymax=ax.get_ylim()[1], color='green', alpha=0.2, label=r'$\mu\pm2\sigma$')
    ax.legend(loc='upper left', fontsize=18, ncols=2)
    

    outFolder = "/t3home/gcelotto/ggHbb/outputs/plots/massSpectrum"
    outName = outFolder + "/dijetMass"    
    if fit:
        outName = outName + "Fit"
    outName = outName + ".png"
    print("Saving in ", outName)
    hep.cms.label(lumi=round(float(currentLumi), 4), ax=ax)
    #ax.text(s="%.2f fb$^{-1}$ (13 TeV)"%currentLumi, x=1.00, y=1.02,  ha='right', transform=ax.transAxes, fontsize=16)
    fig.savefig(outName, bbox_inches='tight')

if __name__=="__main__":
    fit      = int(sys.argv[1]) if len(sys.argv)>1 else True
    realFiles= int(sys.argv[2]) if len(sys.argv)>2 else -1
    plotDijetMass(fit, realFiles=realFiles)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.stats import norm, crystalball
from scipy.integrate import quad
from utilsForPlot import loadData, loadDataOnlyMass, getXSectionBR, loadDataOnlyFeatures, loadRoot, loadParquet, loadDask
import sys, glob
import mplhep as hep
hep.style.use("CMS")

def plotDijetMass(log = False, fit = True, realFiles=1):
        # Loading files
    signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/**"
    realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/**"
    ZJetsToQQPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets"
    columnsToRead =['dijet_mass', 'sf', 'jet1_pt', 'jet2_pt', 'jet2_btagDeepFlavB', 'jet2_qgl', 'ht', 'dijet_pt', 'dijet_dR', 'dijet_dPhi', 'muon_pfRelIso03_all', 'muon_pt']
    
    signal, realData, numEventsTotal= loadParquet(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=-1, nRealDataFiles=realFiles, columns=columnsToRead, returnNumEventsTotal=True)
    

    dfs = [ pd.read_parquet(glob.glob(ZJetsToQQPath+"/ZJetsToQQ_HT-200to400/*.parquet"), columns=columnsToRead),
            pd.read_parquet(glob.glob(ZJetsToQQPath+"/ZJetsToQQ_HT-400to600/*.parquet"), columns=columnsToRead),
            pd.read_parquet(glob.glob(ZJetsToQQPath+"/ZJetsToQQ_HT-600to800/*.parquet"), columns=columnsToRead),
            pd.read_parquet(glob.glob(ZJetsToQQPath+"/ZJetsToQQ_HT-800toInf/*.parquet"), columns=columnsToRead)]
        
        

    df_processes = pd.read_csv("/t3home/gcelotto/ggHbb/bkgEstimation/processes.csv")
    miniDf = pd.read_csv("/t3home/gcelotto/ggHbb/abcd/output/miniDf.csv")
    
    xsectionZ = [df_processes[(df_processes.process=='ZJetsToQQ_200to400')].xsection.values[0],
                    df_processes[(df_processes.process=='ZJetsToQQ_400to600')].xsection.values[0],
                    df_processes[(df_processes.process=='ZJetsToQQ_600to800')].xsection.values[0],
                    df_processes[(df_processes.process=='ZJetsToQQ_800toInf')].xsection.values[0]]
    
    miniZ = [miniDf[(miniDf.process=="ZJetsToQQ_200to400") ].numEventsTotal.sum(),
                miniDf[(miniDf.process=="ZJetsToQQ_400to600") ].numEventsTotal.sum(),
                miniDf[(miniDf.process=="ZJetsToQQ_600to800") ].numEventsTotal.sum(),
                miniDf[(miniDf.process=="ZJetsToQQ_800toInf") ].numEventsTotal.sum()]
    
    totalDataFlat = len(realData)

    def cut(data, feature, min, max):
        newData = []
        for df in data:
            if min is not None:
                df = df[df[feature] > min]
            if max is not None:
                df = df[df[feature] < max]
            newData.append(df)
        return newData
        
    signal, realData = cut([signal, realData], 'jet1_pt', 20, None)
    signal, realData = cut([signal, realData], 'jet2_pt', 20, None)
    
    stringToPrint=''
    columnsToDrop = [x for x in columnsToRead if x not in ['sf', 'dijet_mass']]
    signal, realData = signal.drop(columns=columnsToDrop), realData.drop(columns=columnsToDrop)
        
    dfs = cut(dfs, 'jet1_pt', 20, None)
    dfs = cut(dfs, 'jet2_pt', 20, None)
    for df in dfs:
        df.drop(columns=columnsToDrop)

    lumiPerEvent = np.load("/t3home/gcelotto/ggHbb/outputs/lumiPerEvent.npy")
    
    x1_sb, x2_sb  = 125.09 - 2*17, 125.09 + 2*17
    maskSignal = (signal.dijet_mass>x1_sb) & (signal.dijet_mass<x2_sb)
    maskData = (realData.dijet_mass>x1_sb) & (realData.dijet_mass<x2_sb)
    

    S = np.sum(signal.dijet_mass[maskSignal])*lumiPerEvent*totalDataFlat/numEventsTotal*getXSectionBR()*1000
    B = np.sum(maskData)
    print("Signal 2sigma", S)
    print("Data 2sigma", B)
    print("Sig", S/np.sqrt(B)*np.sqrt(41.6/(lumiPerEvent*totalDataFlat)))

        
        
    realFiles = 1017 if realFiles==-1 else realFiles
    currentLumi = lumiPerEvent*totalDataFlat
    correctionSignal = 1/numEventsTotal*getXSectionBR()*currentLumi*1000
    
    correctionData = 1
    visibilityNonLog = 10
    visibilityNonLogForZ = 1
                  


    # Plot
    fig, ax = plt.subplots(figsize=(6, 7))
    x1, x2, nbin = 0, 300, 101
    ax.set_xlim(x1, x2)
    bins = np.linspace(x1, x2, nbin)
    if log:
        ax.set_yscale('log')

    # Prepare the points
        # Z counts
    
    signalCounts = np.histogram (np.clip(signal.dijet_mass, bins[0], bins[-1]), weights=signal.sf, bins=bins)[0]
    realDataCounts = np.histogram(np.clip(realData, bins[0], bins[-1]), bins=bins)[0]
    realDataCountsErr= np.sqrt(realDataCounts)*correctionData
    signalCountsErr= np.sqrt(signalCounts)*correctionSignal if log==True else np.sqrt(signalCounts)*correctionSignal*visibilityNonLog
    
    ZJetsCounts = np.zeros(len(bins)-1)
    for idx in range(len(dfs)):
        print("df ", np.sum(np.histogram(np.clip(dfs[idx].dijet_mass, bins[0], bins[-1]), weights=dfs[idx].sf, bins=bins)[0] * xsectionZ[idx] / miniZ[idx] * 1000 * currentLumi))
        ZJetsCounts = ZJetsCounts + np.histogram(np.clip(dfs[idx].dijet_mass, bins[0], bins[-1]), weights=dfs[idx].sf, bins=bins)[0] * xsectionZ[idx] / miniZ[idx] * 1000 * currentLumi

        
    #Normalize data
    signalCounts = signalCounts*correctionSignal if log==True else signalCounts*correctionSignal*visibilityNonLog
    realDataCounts = realDataCounts*correctionData
    
    print(np.sum(signalCounts))
    # Plot data
    #ax.hist(bins[:-1], bins=bins, weights=realDataCounts, color='red', histtype=u'step', label='BParking Data')
    ax.hist(bins[:-1], bins=bins, weights=signalCounts, color='blue', histtype=u'step', label='Signal' if log is False else r'Signal $\times 10^{%d} \times$ SFs'%(np.log10(visibilityNonLog)))
    ax.hist(bins[:-1], bins=bins, weights=ZJetsCounts, color='green', histtype=u'step', label='ZJets' if log is False else r'ZJets $\times 10^{%d} \times$ SFs'%(np.log10(visibilityNonLogForZ)))
    ax.set_ylim(ax.get_ylim()[0]+0.01, ax.get_ylim()[1]*1.3)
    
    # Fit
    if (fit):
        x_data_tot = (bins[:-1]+bins[1:])/2
        
        
        x1_fr, x2_fr = 90, 180
        gauss=False
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

            result, abserr= quad(crystal_ball_function, x1_fr, x2_fr, args=(amplitude_fit, mean_fit, sigma_fit, alpha_fit, n_fit))[:2]

        ax.plot(x_data_tot, y_fit, label='Fitted Curve', color='red')

        # Second plot

        #residuals = (signalCounts-y_fit)/(signalCountsErr+0.000001)
        #ax2.errorbar(x_data_tot[fit_region], residuals[fit_region], yerr=1, color='black', linestyle=' ', marker='o')


        for par, var in zip(params, np.diag(covariance)):
            print(par, " +- ", np.sqrt(var))
        #ax.text(s=r"N$=%d\pm%d$"%(result/(bins[1]-bins[0]), abserr/(bins[1]-bins[0])), x=0.04, y=0.95,  ha='left', transform=ax.transAxes, fontsize=12)
        ax.text(s=r"Jet p$_\mathrm{T}>20$ GeV",                                        x=0.05, y=0.75,  ha='left', transform=ax.transAxes, fontsize=18)
        if stringToPrint:
            ax.text(s=stringToPrint,                                                        x=0.05, y=0.55,  ha='left', transform=ax.transAxes, fontsize=18)
        ax.text(s=r"$\mu=%.2f\pm%.2f$"%(mean_fit, np.sqrt(covariance[1, 1])),          x=0.96, y=0.7,  ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s=r"$\sigma=%.2f\pm%.2f$"%(sigma_fit, np.sqrt(covariance[2, 2])),      x=0.96, y=0.65,  ha='right', transform=ax.transAxes, fontsize=12)
        if gauss is False:
            pass
            #ax.text(s=r"$\alpha=%.2f\pm%.2f$"%(alpha_fit, np.sqrt(covariance[3, 3])),      x=0.96, y=0.65,  ha='right', transform=ax.transAxes, fontsize=12)
            #ax.text(s=r"$\mathrm{n}=%.1f\pm%.1f$"%(n_fit, np.sqrt(covariance[4, 4])),      x=0.96, y=0.6,  ha='right', transform=ax.transAxes, fontsize=12)
        chi2 = np.sum(((y_fit[fit_region]-y_fr)/signalCountsErr[fit_region])**2)

        ndof = (len(y_fr)-len(params))
        ax.text(s=r"$\chi^2$/n$_\mathrm{dof}=%d/%d$"%(chi2, ndof),                      x=0.96, y=0.6,  ha='right', transform=ax.transAxes, fontsize=12)



        # EXPECTED SIGNIFICANCE
            #How many events between 95 and 165 GeV
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

        Scommon_exponent = int("{:e}".format(SErr).split('e')[1])
        Bcommon_exponent = int("{:e}".format(BErr).split('e')[1])

        Scommon_exponent = 0 if Scommon_exponent<0 else Scommon_exponent
        if Scommon_exponent==0:
            ax.text(s=r"S(2$\sigma$) = %d $\pm$ %d"%(S/(10**Scommon_exponent), SErr/(10**Scommon_exponent)),                                        x=0.96, y=0.45,     ha='right', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(s=r"S(2$\sigma$) = (%d $\pm$ %d) $\times e%d$"%(S/(10**Scommon_exponent), SErr/(10**Scommon_exponent), Scommon_exponent),       x=0.96, y=0.45,     ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s=r"B(2$\sigma$) = (%d $\pm$ %d) $\times 10^{%d}$"%(B/(10**Bcommon_exponent), BErr/(10**Bcommon_exponent), Bcommon_exponent),       x=0.96, y=0.4,      ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.3f"%(S/(np.sqrt(B))),                                                                                  x=0.96, y=0.35,     ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s="S/B = %.1e"%(S/B),                                                                                                               x=0.96, y=0.3,      ha='right', transform=ax.transAxes, fontsize=12)
        # @ Full lumi
        ax.text(s="@Full BP Lumi (41.6 fb$^{-1}$)",                                                                                                 x=0.96, y=0.17,     ha='right', transform=ax.transAxes, fontsize=12)
        ax.text(s="S/$\mathrm{\sqrt{B}}$ = %.2f"%(S*np.sqrt(41.6/currentLumi)/(np.sqrt(B))),                                                        x=0.96, y=0.12,     ha='right', transform=ax.transAxes, fontsize=12)

        ax.text(s="Yields",                                                        x=0.04, y=0.45,     ha='left', transform=ax.transAxes, fontsize=12)
        ax.text(s="H : %d"%round(np.sum(signal.sf)*correctionSignal),  x=0.04, y=0.4,     ha='left', transform=ax.transAxes, fontsize=12)
        ax.text(s="Z : %d"%round(np.sum(ZJetsCounts)),                             x=0.04, y=0.35,     ha='left', transform=ax.transAxes, fontsize=12)


    # End of Fit
    #ax2.hlines(y=0, xmin=x1, xmax=x2, color='black')
    #ax2.set_ylim(-5.9, 5.9)
    ax.set_xlabel("Dijet Mass [GeV]", fontsize=14)

    #for i in range(len(bins)-1):
    #    if i ==0 :
    #        rect = patches.Rectangle((bins[i], realDataCounts[i] - realDataCountsErr[i]),
    #                bins[i+1]-bins[i], 2 *  realDataCountsErr[i],
    #                linewidth=0, edgecolor='red', facecolor='none', hatch='///') #label='Uncertainty')
    #    else:
    #        rect = patches.Rectangle((bins[i], realDataCounts[i] - realDataCountsErr[i]),
    #                bins[i+1]-bins[i], 2 *  realDataCountsErr[i],
    #                linewidth=0, edgecolor='red', facecolor='none', hatch='///')
    #    ax.add_patch(rect)

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
        ax.vlines(x=[x1_fr, x2_fr], ymin=0, ymax=ax.get_ylim()[1], color='blue', alpha=0.2, label='Fit region')
        ax.vlines(x=[x1_sb, x2_sb], ymin=0, ymax=ax.get_ylim()[1], color='green', alpha=0.2, label=r'$\mu\pm2\sigma$')
    ax.legend(loc='upper left', fontsize=15, ncols=2)

    ax.set_ylabel("Events / %.1f GeV"%(bins[1]-bins[0]), fontsize=16)

    
    outFolder = "/t3home/gcelotto/ggHbb/outputs/plots/massSpectrum"
    outName = outFolder + "/HiggsZmass_bb"
    if fit:
        outName = outName + "Fit"
    if log:
        outName = outName + "Log"
    outName = outName + ".png"
    print("Saving in ", outName)
    print(log, outName)
    ax.text(s="%.2f fb$^{-1}$ (13 TeV)"%currentLumi, x=1.00, y=1.02,  ha='right', transform=ax.transAxes, fontsize=16)
    fig.savefig(outName, bbox_inches='tight')

if __name__=="__main__":

    
    log      = int(sys.argv[1]) if len(sys.argv)>1 else False
    fit      = int(sys.argv[2]) if len(sys.argv)>2 else True
    realFiles= int(sys.argv[3]) if len(sys.argv)>3 else -1
    plotDijetMass(log, fit, realFiles=realFiles)
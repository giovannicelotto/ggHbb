import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import mplhep as hep
hep.style.use("CMS")

def doPlots(Xtrain, Ytrain, YPredTrain, Wtrain, Xtest, Ytest, YPredTest, Wtest, outFolder, logging):
    fig, ax = plt.subplots(1, 1)
    thresholds = np.linspace(0, 1, 1000)
    labels = ['Data', 'ggH', 'ZJets']
    for classIdx in range(1, 3):
        # maskClass : events in the class 1 (ggH) or 2 (ZJets)
        # roc for multiClass: 2 signals, we scan the threshold for Z score and ggH score and see how much Z, QCD and ggH, QCD we retain
        maskClass = Ytest[:,classIdx]>0.99
        maskQCD = Ytest[:,0]>0.99
        maskClassTrain = Ytrain[:,classIdx]>0.99
        maskQCDTrain = Ytrain[:,0]>0.99
        logging.info("%s events in class %d"%(len(Ytest[maskClass]), classIdx))

        tpr, fpr = [], []
        tprTrain, fprTrain = [], []
        for t in thresholds:
            tpr.append(Wtest[(YPredTest[:,classIdx] > t) & (maskClass)].sum()/Wtest[maskClass].sum())
            fpr.append(Wtest[(YPredTest[:,classIdx] > t) & (maskQCD)].sum()/Wtest[maskQCD].sum())

            tprTrain.append(Wtrain[(YPredTrain[:,classIdx]> t) & (maskClassTrain)].sum()/Wtrain[maskClassTrain].sum())
            fprTrain.append(Wtrain[(YPredTrain[:,classIdx]> t) & (maskQCDTrain)].sum()/Wtrain[maskQCDTrain].sum())

        
        from scipy.integrate import simpson
        from sklearn.metrics import auc as auc_sk
        
        auc = auc_sk(fpr, tpr)
        auc_train = auc_sk(fprTrain, tprTrain)
        ax.plot(fpr, tpr, marker='o', markersize=1, label='%s Test AUC %.2f'%(labels[classIdx], auc))
        ax.plot(fprTrain, tprTrain, linestyle='dotted', label='%s Train AUC %.2f'%(labels[classIdx], auc_train))
        ax.plot(thresholds, thresholds, linestyle='dotted', color='green')

        ax.grid(True)
        ax.set_ylabel("Signal Efficiency")
        ax.set_xlabel("QCD Efficiency")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        

    handles, newLabels = ax.get_legend_handles_labels()
    handles = [handles[i] for i in [1, 3, 0, 2]]
    newLabels = [newLabels[i] for i in [1, 3, 0, 2]]
    ax.legend(handles, newLabels, ncols=2)
    hep.cms.label(ax=ax)
    fig.savefig(outFolder +"/performance/roc.png", bbox_inches='tight')
    plt.close('all')


# ggH score scan
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    maskQCD = Ytest[:,0]==1
    #maskQCD = maskQCD.reindex(Xtest.index, fill_value=False)
    maskQCDTrain = Ytrain[:,0]>0.99
    print("Lenghts of masks")
    print(len(YPredTest), len(maskQCD))
    #logging.info("%s events in class %d"%(len(Ytest[maskClass]), classIdx))    
    bins = np.linspace(0, 200, 100)
    t = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for i in range(len(t)):
            mask_scan = (YPredTest[:,1]>t[i]) 
            combinedMask = (maskQCD) & (mask_scan)
            ax.hist(Xtest.dijet_mass[combinedMask], bins=bins, weights=Wtest[combinedMask], label='ggH score >%.1f'%t[i], histtype=u'step', density=True)[0]
            #ax.hist(Xtest.dijet_mass[combinedMask], bins=bins, label='ggH score >%.1f'%t[i], histtype=u'step')[0]
    ax.legend()
    print(Wtest[combinedMask][:10], Wtest[combinedMask].mean(), Wtest[combinedMask].std())
    ax.set_title("Dijet Mass : ggH score scan")
    fig.savefig(outFolder + "/performance/ggHScoreScan.png", bbox_inches='tight')




# confusion matrix
    ax.clear()
    confusionM = np.ones((3, 3))
    for trueLabel in range(3):
        # true label
        maskClass = Ytest[:,trueLabel]>0.99
        den = Wtest[maskClass].sum()
        for predictedLabel in range(3):

            num = (Wtest[(np.argmax(YPredTest, axis=1) == predictedLabel) & (maskClass)]).sum()
            confusionM[trueLabel, predictedLabel] = num/den
    logging.info("Confusion matrix : \n%s"%str(confusionM))
    im = ax.matshow(confusionM, cmap=plt.cm.jet, alpha=0.7, norm=colors.LogNorm(vmin=0.01, vmax=1))
    for y in range(3):
        for x in range(3):
            plt.text(x , y , '%.2f' % confusionM[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16
                    )
    #ax.set_xlabel("Generated (bin number)", fontsize=18)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_xticks([0, 1, 2], ['Data', 'ggH', 'ZJets'])
    ax.set_yticks([0, 1, 2], ['Data', 'ggH', 'ZJets'])
    #ax.set_xlim(0,1)
    #ax.set_ylim(0,1)
        
    #ax.legend()
    hep.cms.label(ax=ax)
    fig.savefig(outFolder +"/performance/confusionMatrix.png", bbox_inches='tight')
    plt.close('all')

# want to see if the dijet mass for qcd is smoothly falling.
# take data events with high ggH score
    fig, ax = plt.subplots(1, 1)
    Xtest=Xtest.reset_index(drop=True)
    bins=np.linspace(0, 200, 40)

    ax.hist(Xtest[(Ytest[:,0]==1)].dijet_mass, bins=bins, weights=Wtest[(Ytest[:,0]==1)], density=True, histtype=u'step')
    ax.hist(Xtest[(Ytest[:,1]==1)].dijet_mass, bins=bins, weights=Wtest[(Ytest[:,1]==1)], density=True, histtype=u'step')
    ax.hist(Xtest[(Ytest[:,2]==1)].dijet_mass, bins=bins, weights=Wtest[(Ytest[:,2]==1)], density=True, histtype=u'step')
    fig.savefig(outFolder +"/performance/dijetMass.png", bbox_inches='tight')
    plt.close('all')

# outputTrainTest
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bins= np.linspace(0, 1, 10)
    xlabels = ['Data score', 'ggH score', 'ZJets score']
    for idx, ax in enumerate(axes):
        mask0 = Ytrain[:,0]==1   #real class 0 
        mask1 = Ytrain[:,1]==1
        mask2 = Ytrain[:,2]==1
        counts0 = np.histogram(YPredTrain[mask0][:,idx], bins=bins, weights=Wtrain[mask0])[0]
        counts1 = np.histogram(YPredTrain[mask1][:,idx], bins=bins, weights=Wtrain[mask1])[0]
        counts2 = np.histogram(YPredTrain[mask2][:,idx], bins=bins, weights=Wtrain[mask2])[0]
        counts0, counts1, counts2, = counts0/np.sum(counts0), counts1/np.sum(counts1), counts2/np.sum(counts2)
        ax.hist(bins[:-1], bins=bins, weights=counts0, label=labels[0]+" train", histtype=u'step', color='C0')
        ax.hist(bins[:-1], bins=bins, weights=counts1, label=labels[1]+" train", histtype=u'step', color='C1')
        ax.hist(bins[:-1], bins=bins, weights=counts2, label=labels[2]+" train", histtype=u'step', color='C2')
        
        mask0 = Ytest[:,0]>0.99
        mask1 = Ytest[:,1]>0.99
        mask2 = Ytest[:,2]>0.99
        counts0 = np.histogram(YPredTest[mask0][:,idx], bins=bins, weights=Wtest[mask0])[0]
        counts1 = np.histogram(YPredTest[mask1][:,idx], bins=bins, weights=Wtest[mask1])[0]
        counts2 = np.histogram(YPredTest[mask2][:,idx], bins=bins, weights=Wtest[mask2])[0]
        counts0, counts1, counts2, = counts0/np.sum(counts0), counts1/np.sum(counts1), counts2/np.sum(counts2)
        ax.errorbar((bins[1:]+bins[:-1])/2, counts0, label=labels[0] + " test", alpha=1, marker = 'o', linestyle='none', color='C0')
        ax.errorbar((bins[1:]+bins[:-1])/2, counts1, label=labels[1] + " test", alpha=1, marker = 'o', linestyle='none', color='C1')
        ax.errorbar((bins[1:]+bins[:-1])/2, counts2, label=labels[2] + " test", alpha=1, marker = 'o', linestyle='none', color='C2')

        ax.set_xlabel(xlabels[idx])
        ax.set_yscale('log')
    
    axes[2].legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.5)
    fig.savefig(outFolder +"/performance/outputTrainTest.png", bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(1, 1)
    print("\n\n\n")
    Xtrain=Xtrain.reset_index(drop=True)
    bins=np.linspace(0, 2, 20)
    counts_Datatrain =  np.histogram(np.clip(Xtrain[Ytrain[:,0]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_ggHtrain =   np.histogram(np.clip(Xtrain[Ytrain[:,1]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_Ztrain =     np.histogram(np.clip(Xtrain[Ytrain[:,2]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_Datatest =   np.histogram(np.clip(Xtest[Ytest[:,0]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_ggHtest =    np.histogram(np.clip(Xtest[Ytest[:,1]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_Ztest =      np.histogram(np.clip(Xtest[Ytest[:,2]==1].sf, bins[0], bins[-1]), bins=bins)[0]
    counts_Datatrain = counts_Datatrain/np.sum(counts_Datatrain)
    counts_ggHtrain = counts_ggHtrain/np.sum(counts_ggHtrain)
    counts_Ztrain = counts_Ztrain/np.sum(counts_Ztrain)
    counts_Datatest = counts_Datatest/np.sum(counts_Datatest)
    counts_ggHtest = counts_ggHtest/np.sum(counts_ggHtest)
    counts_Ztest = counts_Ztest/np.sum(counts_Ztest)

    ax.hist(bins[:-1], bins=bins, weights=counts_Datatrain, histtype=u'step', linewidth=2 ,label='Data train %d'%(Xtrain[Ytrain[:,0]==1].sf.sum()))    
    ax.hist(bins[:-1], bins=bins, weights=counts_ggHtrain, histtype=u'step', linewidth=2 ,label='ggH train %d'%(Xtrain[Ytrain[:,1]==1].sf.sum()))
    ax.hist(bins[:-1], bins=bins, weights=counts_Ztrain, histtype=u'step', linewidth=2 ,label='Z train %d'%(Xtrain[Ytrain[:,2]==1].sf.sum()))
    ax.hist(bins[:-1], bins=bins, weights=counts_Datatest, histtype=u'step', linewidth=2 ,label='Data test %d'%(Xtest[Ytest[:,0]==1].sf.sum()))
    ax.hist(bins[:-1], bins=bins, weights=counts_ggHtest, histtype=u'step', linewidth=2 ,label='ggH test %d'%(Xtest[Ytest[:,1]==1].sf.sum()))
    ax.hist(bins[:-1], bins=bins, weights=counts_Ztest, histtype=u'step', linewidth=2 ,label='Z test %d'%(Xtest[Ytest[:,2]==1].sf.sum()))
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(outFolder +"/performance/weightsDistribution.png")


    return
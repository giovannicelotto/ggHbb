import shap
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import numpy as np
import pandas as pd
import glob
import sys
from functions import getXSectionBR
import mplhep as hep
hep.style.use("CMS")
def doPlotLoss(fit, outName, earlyStop, patience):

    # "Loss"
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    ax.plot(fit.history['loss'])
    ax.plot(fit.history['val_loss'])
    #ax.plot(fit.history['accuracy'])
    #ax.plot(fit.history['val_accuracy'])
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    
    # plt.yscale('log')
    ax.set_ylim(ymax = max(min(fit.history['loss']), min(fit.history['val_loss']))*1.4, ymin = min(min(fit.history['loss']),min(fit.history['val_loss']))*0.9)
    #ax.set_ylim(ymin= min(min(fit.history['loss']),min(fit.history['val_loss']))*0.9, ymax=max(min(fit.history['accuracy']), min(fit.history['val_accuracy']))*1.4)
    #ax.set_ylim(0, 1)
    ymax = 1#min(fit.history['val_loss'])
    ymin = 0#plt.ylim()[0]
    plt.arrow(x=earlyStop.stopped_epoch-patience+1, y=ymax, dx=0, dy=ymin-ymax, length_includes_head=True, head_length=0.033*(ymin-ymax))
    #plt.legend(['Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy'], loc='upper right')
    plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
    plt.savefig(outName)
    plt.cla()
    print("Saved loss function ", outName)


def roc(thresholds, signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outName):

    tpr, tpr_train = [], []
    fpr, fpr_train = [], []
    for t in thresholds:
        tpr.append(np.sum(signal_predictions > t)/len(signal_predictions))
        fpr.append(np.sum(realData_predictions > t)/len(realData_predictions))
        tpr_train.append(np.sum(signalTrain_predictions > t)/len(signalTrain_predictions))
        fpr_train.append(np.sum(realDataTrain_predictions > t)/len(realDataTrain_predictions))
    tpr, fpr =np.array(tpr), np.array(fpr)
    tpr_train, fpr_train =np.array(tpr_train), np.array(fpr_train)
    # auc
    auc = -simpson(tpr, fpr)
    auc_train = -simpson(tpr_train, fpr_train)
    print("AUC : ", auc)


    fig, ax = plt.subplots(1, 1)
    ax.plot(fpr, tpr, marker='o', markersize=1, label='test')
    ax.plot(fpr_train, tpr_train, marker='o', markersize=1, label='train')
    ax.plot(thresholds, thresholds, linestyle='dotted', color='green')
    
    ax.grid(True)
    ax.set_ylabel("Signal Efficiency")
    ax.set_xlabel("Background Efficiency")
    ax.set_xlim(1e-5,1)
    ax.set_ylim(1e-1,1)
    ax.text(x=0.95, y=0.32, s="AUC Test : %.3f"%auc, ha='right')
    ax.text(x=0.95, y=0.28, s="AUC Train: %.3f"%auc_train, ha='right')
    ax.legend()
    fig.savefig(outName, bbox_inches='tight')
    plt.close('all')

def WorkingPoint(signal_predictions, realData_predictions, SFtest, outName):
    cuts = [0.00001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.6, 0.65, 0.7, 0.75, 0.775, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,0.88, 0.9, 0.95, 0.99, 0.995, 0.999]
    bkgRetained = []
    signalRetained = []
    for c in cuts:
        print("BKG retained statistics : ", np.sum(realData_predictions>c))
        bkgRetained.append(np.sum(realData_predictions>c)/len(realData_predictions))
        signalRetained.append(np.sum(SFtest[signal_predictions>c])/np.sum(SFtest))
        print(c, bkgRetained[-1], signalRetained[-1], signalRetained[-1]/np.sqrt(bkgRetained[-1]))
    fig, ax = plt.subplots(1, 1)
    ax.plot(cuts, signalRetained,  label='signalRetained')
    ax.plot(cuts, bkgRetained, label='bkgRetained')
    ax.set_ylabel('Efficiency')
    ax2 = ax.twinx()
    ax2.plot(cuts, signalRetained/np.sqrt(bkgRetained), label='significance gain', color='green')
    ax2.set_ylabel('Gain in significance', color='green')
    ax.legend(loc='upper center')
    ax.set_xlabel("NN output")
    fig.savefig(outName, bbox_inches='tight')
    plt.close('all')

def massSpectrum(Xtest, Ytest, y_predict, SFtest, hp, numEventsTotalTest, outName):
    fig, axes = plt.subplots(3, 3, constrained_layout=True, figsize=(20, 12))
    #fig.subplots_adjust(wspace=0.25)
    bins=np.linspace(0, 300, 101)
    totalDataFlat_test = np.load("/t3home/gcelotto/ggHbb/outputs/totalDataFlat_test.npy")
    lumiPerEvent = np.load("/t3home/gcelotto/ggHbb/outputs/lumiPerEvent.npy")
    #assert len(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ggH2023Dec06/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231206_105206/flatData/others/*.parquet"))==192
    
    print("Current lumi", lumiPerEvent*totalDataFlat_test)
    visibilityFactor= 100
    workingPoints = [0, 0.25, 0.5,
                     0.8, 0.9, 0.95,
                     0.98, 0.99, 0.995]

    print(type(Xtest.dijet_mass), type(Ytest.label), type(y_predict.label))
    assert len(SFtest)==len(Xtest)
    for idx, ax in enumerate(axes.reshape(-1)):

        signalCounts = np.histogram(Xtest.dijet_mass[(Ytest.label==1) & (y_predict.label>workingPoints[idx])], bins=bins, weights=SFtest[(Ytest.label==1) & (y_predict.label>workingPoints[idx])])[0]*lumiPerEvent*totalDataFlat_test/numEventsTotalTest*getXSectionBR()*1000*visibilityFactor
        realDataCounts = np.histogram(Xtest.dijet_mass[(Ytest.label==0) & (y_predict.label>workingPoints[idx])], bins=bins)[0]
        ax.hist(bins[:-1], bins=bins, weights=signalCounts, histtype=u'step', color='blue', label='MC ggHbb x %d'%visibilityFactor)
        ax.hist(bins[:-1], bins=bins, weights=realDataCounts, histtype=u'step', color='red', label='BParking Data')


        x1_sb, x2_sb  = 123.11 - 2*17, 123.11 + 2*17
        maskSignal = (Xtest[Ytest.label==1].dijet_mass>x1_sb) & (Xtest[Ytest.label==1].dijet_mass<x2_sb)
        maskSignal = (maskSignal) & (y_predict.label[Ytest.label==1]>workingPoints[idx])
        maskData = (Xtest[Ytest.label==0].dijet_mass>x1_sb) & (Xtest[Ytest.label==0].dijet_mass<x2_sb)
        maskData = (maskData) & (y_predict.label[Ytest.label==0]>workingPoints[idx])
        S = np.sum(SFtest[(Ytest.label==1) & (maskSignal)])*lumiPerEvent*totalDataFlat_test/numEventsTotalTest*getXSectionBR()*1000
        B = np.sum(maskData)
        print("Signal 2sigma", S)
        print("Data 2sigma", B)
        sig=S/np.sqrt(B)*np.sqrt(41.6/(lumiPerEvent*totalDataFlat_test))
        print("Sig", sig)

        ax.text(x=0.9, y=0.5, s="Cut: %.3f"%(round(workingPoints[idx], 3)), transform=ax.transAxes, ha='right')
        ax.text(x=0.9, y=0.4, s="Sig: %.2f"%(round(sig, 2)), transform=ax.transAxes, ha='right')
        ax.text(x=0.9, y=0.3, s="S/B: %.2f"%(round(S/np.sqrt(B), 2)), transform=ax.transAxes, ha='right')
        ax.set_ylabel('Events/GeV')
        ax.legend(loc='upper right')
        ax.set_xlabel("Dijet mass [GeV]")
        fig.savefig(outName, bbox_inches='tight')
    plt.close('all')


def NNoutputs(signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions, outName):
    #signal_predictions, realData_predictions, signalTrain_predictions, realDataTrain_predictions = np.arctanh(signal_predictions), np.arctanh(realData_predictions), np.arctanh(signalTrain_predictions), np.arctanh(realDataTrain_predictions)
    fig, ax = plt.subplots(1, 1)
    bins=np.linspace(0, 1, 20)
    sig_test_counts = np.histogram(signal_predictions, bins=bins)[0]
    bkg_test_counts = np.histogram(realData_predictions, bins=bins)[0]
    sig_train_counts = np.histogram(signalTrain_predictions, bins=bins)[0]
    bkg_train_counts = np.histogram(realDataTrain_predictions, bins=bins)[0]
    sig_train_counts_err = np.sqrt(sig_train_counts)
    bkg_train_counts_err = np.sqrt(bkg_train_counts)
    sig_test_counts, bkg_test_counts, sig_train_counts, bkg_train_counts, sig_train_counts_err, bkg_train_counts_err = sig_test_counts/np.sum(sig_test_counts), bkg_test_counts/np.sum(bkg_test_counts), sig_train_counts/np.sum(sig_train_counts), bkg_train_counts/np.sum(bkg_train_counts), sig_train_counts_err/np.sum(sig_train_counts), bkg_train_counts_err/np.sum(bkg_train_counts)
    ax.hist(bins[:-1], bins=bins, weights=sig_test_counts, alpha=0.3, label='signal test')
    ax.hist(bins[:-1], bins=bins, weights=bkg_test_counts, alpha=.3, label='bkg test')
    ax.errorbar(x=(bins[1:]+bins[:-1])/2, y=sig_train_counts, yerr=sig_train_counts_err, linestyle='none', label='signal train')
    ax.errorbar(x=(bins[1:]+bins[:-1])/2, y=bkg_train_counts, yerr=bkg_train_counts_err, linestyle='none', label='bkg train')
    ax.legend(loc='upper center')
    ax.set_yscale('log')
    ax.set_xlabel("NN output")
    fig.savefig(outName, bbox_inches='tight')


def getShap(Xtest, model, outName, class_names=['NN output']):
    
    plt.figure()
    max_display = len(Xtest.columns)
    #max_display = 32
    explainer = shap.GradientExplainer(model=model, data=Xtest)
    print("exapliner started")
    shap_values = explainer.shap_values(np.array(Xtest), nsamples=1000)
        #Generate summary plot
    shap.initjs()
    shap.plots.bar(shap_values)
    #shap.summary_plot(shap_values, Xtest, plot_type="bar",
    #                feature_names=Xtest.columns,
    #                max_display=max_display,
    #                plot_size=[15.0,0.4*max_display+1.5],
    #                class_names=class_names,
    #                show=False)
    plt.savefig(outName)

def getShapNew(Xtest, model, outName, nFeatures, class_names=['NN output']):
    featuresForTraining = Xtest.columns.values
    explainer = shap.GradientExplainer(model=model, data=Xtest)

    shap_values = explainer.shap_values(np.array(Xtest))
    print(shap_values)
    # get the contribution for each class by averaging over all the events
    shap_values_average = abs(shap_values).mean(axis=0)

    #find the leading features:
    shap_values_average_sum = shap_values_average.sum(axis=1)
    for idx, feature in enumerate(featuresForTraining):
        print(feature, shap_values_average_sum[idx])
    indices = np.argsort(shap_values_average_sum)[::-1]
    ordered_featuresForTraining = np.array(featuresForTraining)[indices][:nFeatures]
    orderd_shap_values_average = shap_values_average[indices][:nFeatures]

    fig, ax =plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    bins = np.arange(len(ordered_featuresForTraining)+1)



    c0=ax.hist(bins[:-1],bins=bins, width=0.3,color='C0', weights=orderd_shap_values_average[:,0] ,label='Data')[0]
    c1=ax.hist(bins[:-1],bins=bins, width=0.3,color='C1', weights=orderd_shap_values_average[:,1] ,label='Higgs', bottom=c0)[0]
    c2=ax.hist(bins[:-1],bins=bins, width=0.3,color='C2', weights=orderd_shap_values_average[:,2] , label='Z',bottom=c0+c1)[0]
    ax.set_xticks(np.arange(len(ordered_featuresForTraining)), labels=ordered_featuresForTraining)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.legend()
    ax.set_ylabel("Mean(|SHAP|)")
    plt.savefig(outName)